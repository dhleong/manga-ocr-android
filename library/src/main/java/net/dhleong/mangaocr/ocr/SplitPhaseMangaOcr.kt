package net.dhleong.mangaocr.ocr

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.android.gms.tflite.java.TfLite
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import net.dhleong.mangaocr.MangaOcr
import net.dhleong.mangaocr.Vocab
import net.dhleong.mangaocr.hub.HfHubRepo
import net.dhleong.mangaocr.hub.ModelPath
import net.dhleong.mangaocr.onnx.FloatTensor
import net.dhleong.mangaocr.onnx.FloatTensor.Companion.allocateFloatOutputTensor
import net.dhleong.mangaocr.onnx.createSession
import net.dhleong.mangaocr.tflite.await
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.File
import java.nio.LongBuffer

class SplitPhaseMangaOcr(
    private val encoder: Encoder,
    private val decoder: Decoder,
    private val vocab: Vocab,
    private val maxChars: Int = 300,
) : MangaOcr {
    interface Encoder {
        fun encode(bitmap: Bitmap): FloatTensor
    }

    interface Decoder {
        /** @return logits */
        fun decode(
            encoderHiddenStates: FloatTensor,
            tokenIds: LongBuffer,
            tokensCount: Int,
        ): FloatTensor
    }

    private class OnnxEncoder(
        private val encoder: OrtSession,
        private val imageProcessor: net.dhleong.mangaocr.ImageProcessor<OnnxTensor> =
            net.dhleong.mangaocr.ImageProcessor { floats, shape ->
                OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), floats, shape)
            },
    ) : Encoder {
        override fun encode(bitmap: Bitmap): FloatTensor {
            val processed = imageProcessor.preprocess(bitmap)
            val outputs = encoder.run(mapOf("pixel_values" to processed))
            return FloatTensor.from(outputs.get("encoder_hidden_states").get()).assertHasRows()
        }
    }

    private class TfliteEncoder(
        private val encoder: InterpreterApi,
        private val targetWidth: Int = 224,
        private val targetHeight: Int = 224,
    ) : Encoder {
        override fun encode(bitmap: Bitmap): FloatTensor {
            val imageProcessor =
                ImageProcessor
                    .Builder()
                    .add(ResizeOp(targetHeight, targetWidth, ResizeOp.ResizeMethod.BILINEAR))
                    .add(NormalizeOp(127.5f, 127.5f)) // [ -1, 1 ]
                    .build()
            val processed = imageProcessor.process(TensorImage.fromBitmap(bitmap))

            val encoded = encoder.allocateFloatOutputTensor(0)
            encoder.runForMultipleInputsOutputs(
                arrayOf(processed.buffer),
                mapOf(1 to encoded.buffer),
            )
            return encoded
        }
    }

    private class OnnxDecoder(
        private val decoder: OrtSession,
    ) : Decoder {
        override fun decode(
            encoderHiddenStates: FloatTensor,
            tokenIds: LongBuffer,
            tokensCount: Int,
        ): FloatTensor {
            val tokenIdsTensor =
                OnnxTensor.createTensor(
                    OrtEnvironment.getEnvironment(),
                    tokenIds,
                    longArrayOf(1, tokensCount.toLong()),
                )
            val hiddenStatesTensor =
                OnnxTensor.createTensor(
                    OrtEnvironment.getEnvironment(),
                    encoderHiddenStates.buffer,
                    encoderHiddenStates.createLongShape(),
                )
            Log.v("TfliteMangaOcr", "inputInfo=${decoder.inputInfo} ${decoder.inputInfo["encoder_hidden_states"]?.info}")
            val outputs = decoder.run(mapOf("encoder_hidden_states" to hiddenStatesTensor, "input_ids" to tokenIdsTensor))
            return FloatTensor.from(outputs.get("logits").get()).assertHasRows()
        }
    }

    private class TfliteDecoder(
        private val decoder: InterpreterApi,
    ) : Decoder {
        override fun decode(
            encoderHiddenStates: FloatTensor,
            tokenIds: LongBuffer,
            tokensCount: Int,
        ): FloatTensor {
            Log.v("TfliteMangaOcr", "output=${decoder.getOutputTensor(0).name()}")
            val logits = decoder.allocateFloatOutputTensor(0)
            decoder.runForMultipleInputsOutputs(
                arrayOf(encoderHiddenStates.buffer, tokenIds),
                mapOf(0 to logits.buffer),
            )
            return logits
        }
    }

    override suspend fun process(bitmap: Bitmap): Flow<MangaOcr.Result> =
        flow {
            val hiddenStatesTensor =
                withTiming("encode") {
                    encoder.encode(bitmap)
                }

            val tokenIds = LongBuffer.allocate(maxChars)
            tokenIds.put(2) // start token
            val result = StringBuilder()

            for (tokensCount in 1 until maxChars) {
                // Prepare to read the tokenIds:
                tokenIds.flip()

                val logits =
                    withTiming("decode") {
                        // TODO: Is this necessary?
                        hiddenStatesTensor.buffer.limit(hiddenStatesTensor.buffer.capacity())
                        hiddenStatesTensor.buffer.position(0)

                        decoder.decode(hiddenStatesTensor, tokenIds, tokensCount)
                    }
                val maxTokenId = logits.maxValuedIndexInRow(logits.lastRowIndex)
                val token = vocab.lookupToken(maxTokenId)

                if (maxTokenId >= 5) {
                    result.append(token)
                    emit(MangaOcr.Result.Partial(result))
                } else if (maxTokenId == 3) {
                    break
                }

                tokenIds.limit(tokensCount + 1)
                tokenIds.position(tokensCount)
                tokenIds.put(maxTokenId.toLong())
                Log.v("SplitPhaseMangaOcr", "Got token $maxTokenId ($token)")
            }

            emit(MangaOcr.Result.FinalResult(result.toString()))
        }.flowOn(Dispatchers.IO)

    companion object {
        private val TFLITE_MODEL_ENCODER =
            ModelPath(
                path = "manga-ocr.converted.encoder.tflite",
                sha256 = "",
            )

        private val TFLITE_MODEL_DECODER =
            ModelPath(
                path = "manga-ocr.converted.decoder.tflite",
                sha256 = "",
            )

        // NOTE: "converted" is not *really* a proper prefix for these files since it's a
        // pretty straightforward dump to onnx from pytorch but... we'll keep it for historical
        // reasons, I guess (we originally wanted to convert to tflite; see above).
        private val ONNX_MODEL_ENCODER =
            ModelPath(
                path = "manga-ocr.converted.encoder.preprocessed.quant.onnx",
                sha256 = "5766730bc8e894d8f61f8671f59eabe59afe9d614c05d1fcd52b306f15458459",
            )

        private val ONNX_MODEL_DECODER =
            ModelPath(
                path = "manga-ocr.converted.decoder.preprocessed.quant.onnx",
                sha256 = "a9ba33fdf9020b25e227a7b97abab8a5be16872745888eb49b017507efd4b2b5",
            )

        /**
         * Do not use the tflite models. They don't work (and aren't uploaded besides)
         * and are simply kept around in case we want to experiment later
         */
        suspend fun initialize(
            context: Context,
            useTfliteEncoder: Boolean = false,
            useTfliteDecoder: Boolean = false,
        ): MangaOcr =
            coroutineScope {
                val repo = HfHubRepo("dhleong/manga-ocr-android")
                val initialized =
                    if (useTfliteEncoder || useTfliteDecoder) {
                        async { TfLite.initialize(context).await() }
                    } else {
                        async { Unit }
                    }
                val encoder =
                    async {
                        if (useTfliteEncoder) {
                            val path = repo.resolveLocalPath(context, TFLITE_MODEL_ENCODER)
                            TfliteEncoder(buildInterpreter(initialized, path))
                        } else {
                            val path = repo.resolveLocalPath(context, ONNX_MODEL_ENCODER)
                            OnnxEncoder(buildSession(path))
                        }
                    }
                val decoder =
                    async {
                        if (useTfliteDecoder) {
                            val path = repo.resolveLocalPath(context, TFLITE_MODEL_DECODER)
                            TfliteDecoder(buildInterpreter(initialized, path))
                        } else {
                            val path = repo.resolveLocalPath(context, ONNX_MODEL_DECODER)
                            OnnxDecoder(buildSession(path))
                        }
                    }
                val vocab = async { Vocab.fetch(context) }

                SplitPhaseMangaOcr(encoder.await(), decoder.await(), vocab.await())
            }

        private suspend fun buildInterpreter(
            initialized: Deferred<Unit>,
            modelPath: File,
        ): InterpreterApi {
            initialized.await()
            return InterpreterApi.create(
                modelPath,
                InterpreterApi.Options().apply {
                    runtime = InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY
                },
            )
        }

        private suspend fun buildSession(modelPath: File): OrtSession =
            createSession(modelPath) {
                setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)

//                addXnnpack(emptyMap())
//                addNnapi()
                addCPU(true)
                Log.v("ORT", "procs=${Runtime.getRuntime().availableProcessors()}")
                setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(4))

                // This is the recommended xnnpack config, but it's way slower:
//                        addXnnpack(mapOf("intra_op_num_threads" to "4"))
//                        addConfigEntry("session.intra_op.allow_spinning", "0")
//                        setIntraOpNumThreads(1)
            }
    }
}

inline fun <T> withTiming(
    what: String,
    op: () -> T,
): T {
    val start = System.currentTimeMillis()
    val result = op()
    val deltaMs = System.currentTimeMillis() - start
    Log.v("SplitPhaseMangaOcr", "$what($deltaMs ms -> $result)")
    return result
}
