use std::io::Read;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use bytes::Bytes;
use futures::FutureExt;
use futures::task::noop_waker_ref;
use zstd::zstd_safe::WriteBuf;
use arrow_array::*;
use arrow_schema::DataType;
use crate::arrow::async_reader::AsyncFileReader;
use crate::arrow::array_reader::byte_array::ByteArrayDecoderPlain;
use crate::arrow::buffer::offset_buffer::OffsetBuffer;
use crate::basic::{Encoding, PageType};
use crate::column::page::{Page, PageReader};
use crate::column::reader::{ColumnReader, get_column_reader};
use crate::data_type::*;
use crate::encodings::decoding::{Decoder, PlainDecoder};
use crate::file::metadata::ParquetMetaData;
use crate::file::reader::{ChunkReader, Length, SerializedPageReader};

pub async fn get_dictionary_for_pruning<T: AsyncFileReader + Send + 'static>(
    input: &mut T,
    parquet_metadata: &ParquetMetaData,
    row_group_idx: usize,
    col_idx: usize,
) -> crate::errors::Result<Option<ArrayRef>> {

    let row_group_metadata = parquet_metadata.row_group(row_group_idx);
    let column_metadata = row_group_metadata.column(col_idx);

    // check whether dictionary page exists
    if column_metadata
        .page_encoding_stats()
        .iter()
        .flat_map(|stats| stats.iter())
        .all(|stat| stat.page_type != PageType::DICTIONARY_PAGE)
    {
        // no dictionary page
        return Ok(None);
    }

    // check whether dictionary encoded page exists, if so, cannot use for pruning
    if column_metadata
        .page_encoding_stats()
        .iter()
        .flat_map(|stats| stats.iter())
        .any(|stat| {
            !matches!(stat.encoding, Encoding::PLAIN_DICTIONARY | Encoding::RLE_DICTIONARY)
        })
    {
        // at least one non-dictionary encoding is present
        return Ok(None);
    }

    // check whether non-dictionary encoding exists in column meta, if so, cannot use for pruning
    if column_metadata.encodings().contains(&Encoding::PLAIN_DICTIONARY) {
        if column_metadata
            .encodings()
            .iter()
            .any(|enc| {
                !matches!(enc, Encoding::PLAIN_DICTIONARY | Encoding::RLE | Encoding::BIT_PACKED)
            })
        {
            // if remove returned true, PLAIN_DICTIONARY was present, which means at
            // least one page was dictionary encoded and 1.0 encodings are used

            // RLE and BIT_PACKED are only used for repetition or definition levels
            return Ok(None);
        }
    } else {
        // if PLAIN_DICTIONARY wasn't present, then either the column is not
        // dictionary-encoded, or the 2.0 encoding, RLE_DICTIONARY, was used.
        // for 2.0, this cannot determine whether a page fell back without
        // page encoding stats
        return Ok(None);
    }

    let mut page_reader: Box<dyn PageReader> = Box::new(SerializedPageReader::new(
        Arc::new(WrappedChunkReader {
            inner: Arc::new(Mutex::new(input)),
            start: column_metadata.byte_range().0,
            len: column_metadata.byte_range().1,
        }),
        column_metadata,
        row_group_metadata.num_rows() as usize,
        None,
    )?);
    let page_metadata = page_reader.peek_next_page()?;
    if !page_metadata.map(|metadata| metadata.is_dict).unwrap_or(false) {
        return Ok(None);
    }

    macro_rules! get_prim_dict {
        ($r:expr, $parquet_type:ty, $array_type:ty) => {{
            match $r.page_reader.get_next_page()? {
                Some(Page::DictionaryPage { buf, num_values, .. }) => {
                    let mut dict_values = vec![Default::default(); num_values as usize];
                    let mut dict_decoder = PlainDecoder::<$parquet_type>::new(
                        column_metadata.column_descr().type_length());
                    dict_decoder.set_data(buf, num_values as usize)?;
                    dict_decoder.get(&mut dict_values)?;
                    Ok(Some(
                        Arc::new(<$array_type>::from_iter_values(dict_values))
                    ))
                }
                _ => {
                    Ok(None)
                }
            }
        }}
    }

    // safety:
    //  bypass lifetime checking.
    //  page_reader is guarenteed not to be used outside its lifetime.
    let unsafe_page_reader = unsafe {
        std::mem::transmute::<_, Box<dyn PageReader>>(page_reader)
    };
    match get_column_reader(column_metadata.column_descr_ptr(), unsafe_page_reader) {
        ColumnReader::BoolColumnReader(_) => {
            return Ok(None);
        },
        ColumnReader::Int32ColumnReader(mut r) => {
            return get_prim_dict!(r, Int32Type, Int32Array);
        },
        ColumnReader::Int64ColumnReader(mut r) => {
            return get_prim_dict!(r, Int64Type, Int64Array);
        },
        ColumnReader::Int96ColumnReader(_) => {
            return Ok(None);
        },
        ColumnReader::FloatColumnReader(mut r) => {
            return get_prim_dict!(r, FloatType, Float32Array);
        },
        ColumnReader::DoubleColumnReader(mut r) => {
            return get_prim_dict!(r, DoubleType, Float64Array);
        },
        ColumnReader::ByteArrayColumnReader(mut r) => {
            match r.page_reader.get_next_page()? {
                Some(Page::DictionaryPage { buf, num_values, .. }) => {
                    let mut buffer = OffsetBuffer::<i32>::default();
                    let mut decoder = ByteArrayDecoderPlain::new(
                        buf,
                        num_values as usize,
                        Some(num_values as usize),
                        false,
                    );
                    decoder.read(&mut buffer, usize::MAX)?;
                    return Ok(Some(buffer.into_array(None, DataType::Binary)));
                }
                _ => return Ok(None),
            }
        },
        ColumnReader::FixedLenByteArrayColumnReader(_) => {
            return Ok(None);
        }
    }
}

struct WrappedChunkReader<'a, T: AsyncFileReader + Send + 'static> {
    inner: Arc<Mutex<& 'a mut T>>,
    start: u64,
    len: u64,
}
unsafe impl<T: AsyncFileReader + Send + 'static> Sync for WrappedChunkReader<'_, T> {}

impl <T: AsyncFileReader + Send + 'static> Read for WrappedChunkReader<'_, T> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        buf.copy_from_slice(self.get_bytes(0, buf.len())?.as_slice());
        self.start += buf.len() as u64;
        Ok(buf.len())
    }
}

impl<T: AsyncFileReader + Send + 'static> Length for WrappedChunkReader<'_, T> {
    fn len(&self) -> u64 {
        self.len
    }
}
impl <T: AsyncFileReader + Send + 'static> ChunkReader for WrappedChunkReader<'_, T> {
    type T = Self;

    fn get_read(&self, start: u64) -> crate::errors::Result<Self::T> {
        Ok(Self {
            inner: self.inner.clone(),
            start,
            len: self.len,
        })
    }

    fn get_bytes(&self, start: u64, length: usize) -> crate::errors::Result<Bytes> {
        let start = (self.start + start) as usize;
        let end = start + length;
        let mut future = Box::pin(async move {
            let mut inner = self.inner.lock().unwrap();
            inner.get_bytes(start..end).await
        });

        // inner.get_bytes() is sync in blaze, so poll() has no extra cost
        loop {
            match future.poll_unpin(&mut Context::from_waker(noop_waker_ref())) {
                Poll::Ready(bytes_result) => return Ok(bytes_result?),
                Poll::Pending => continue,
            }
        }
    }
}

#[cfg(test)]
mod test {
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};
    use std::ops::Range;
    use std::sync::Arc;
    use bytes::Bytes;
    use futures::future::BoxFuture;
    use arrow_array::Array;
    use arrow_array::cast::as_string_array;
    use arrow_cast::cast;
    use arrow_schema::DataType;
    use crate::arrow::arrow_reader::ArrowReaderOptions;
    use crate::arrow::async_reader::AsyncFileReader;
    use crate::arrow::ParquetRecordBatchStreamBuilder;
    use crate::blaze::get_dictionary_for_pruning;
    use crate::file::footer::{decode_footer, decode_metadata};
    use crate::file::FOOTER_SIZE;
    use crate::file::metadata::ParquetMetaData;
    use crate::file::reader::ChunkReader;

    struct AsyncLocalFileReader(File);
    impl AsyncFileReader for AsyncLocalFileReader {
        fn get_bytes(&mut self, range: Range<usize>) -> BoxFuture<'_, crate::errors::Result<Bytes>> {
            Box::pin(async move {
                // eprintln!("IO get_bytes {:?}", range);
                Ok(self.0.get_bytes(range.start as u64, range.len())?)
            })
        }

        fn get_metadata(&mut self) -> BoxFuture<'_, crate::errors::Result<Arc<ParquetMetaData>>> {
            Box::pin(async move {
                self.0.seek(SeekFrom::End(-(FOOTER_SIZE as i64)))?;

                let mut buf = [0_u8; FOOTER_SIZE];
                self.0.read_exact(&mut buf)?;

                let metadata_len = decode_footer(&buf)?;
                self.0.seek(SeekFrom::End(-(FOOTER_SIZE as i64) - metadata_len as i64))?;

                let mut buf = Vec::with_capacity(metadata_len);
                self.0.try_clone()?.take(metadata_len as _).read_to_end(&mut buf)?;

                let metadata = decode_metadata(&buf)?;
                Ok(Arc::new(metadata))
            })
        }
    }

    #[tokio::test]
    async fn test_get_dictionary() -> crate::errors::Result<()> {
        let input = AsyncLocalFileReader(
            File::open("/Volumes/Workspace/blaze-init/arrow-rs/with-dict-2.parquet")?
        );
        let mut builder = ParquetRecordBatchStreamBuilder::new_with_options(
            input,
            ArrowReaderOptions::new().with_page_index(true),
        ).await?;

        let col_idx = builder
            .schema()
            .fields()
            .iter()
            .position(|f| f.name().eq_ignore_ascii_case("status"))
            .unwrap();

        let dict = get_dictionary_for_pruning(&mut builder.input.0, &builder.metadata, 0, col_idx)
            .await?
            .expect("no dictionary");
        let dict = cast(&dict, &DataType::Utf8)?;
        let dict_strs = as_string_array(&dict);

        eprintln!("got dictionary: {:?}", dict_strs);
        eprintln!("dict bytes: {}", dict_strs.offsets()[dict_strs.len()] - dict_strs.offsets()[0]);
        assert!(!dict.is_empty());
        Ok(())
    }
}
