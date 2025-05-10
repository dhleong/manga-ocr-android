package net.dhleong.mangaocr.ocr

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.android.gms.tflite.java.TfLite
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import net.dhleong.mangaocr.MangaOcr
import net.dhleong.mangaocr.Vocab
import net.dhleong.mangaocr.hub.HfHubRepo
import net.dhleong.mangaocr.onnx.FloatTensor.Companion.allocateFloatOutputTensor
import net.dhleong.mangaocr.tflite.await
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.nio.IntBuffer
import kotlin.system.measureTimeMillis

class TfliteMangaOcr(
    private val encoder: InterpreterApi,
    private val decoder: InterpreterApi,
    private val vocab: Vocab,
    private val maxChars: Int = 300,
    private val targetWidth: Int = 224,
    private val targetHeight: Int = 224,
) : MangaOcr {
    override suspend fun process(bitmap: Bitmap): Flow<MangaOcr.Result> =
        flow {
            val imageProcessor =
                ImageProcessor
                    .Builder()
                    .add(ResizeOp(targetHeight, targetWidth, ResizeOp.ResizeMethod.BILINEAR))
                    .add(NormalizeOp(0f, 255f)) // [ 0, 1 ]
//                    .add(NormalizeOp(127.5f, 127.5f)) // [ -1, 1 ]
                    .build()
            val processed = imageProcessor.process(TensorImage.fromBitmap(bitmap))

            for (i in 0 until encoder.outputTensorCount) {
                val t = encoder.getOutputTensor(i)
                Log.v("TfliteDetector", "encoder($i): ${t.name()} / ${t.shape().toList()}")
            }

            val encoded = encoder.allocateFloatOutputTensor(1)
            Log.v("TfliteDetector", "encode(...): ${encoded.name}=${encoded.buffer.array().toList()}")
            val encodeTimeMs =
                measureTimeMillis {
                    encoder.runForMultipleInputsOutputs(
                        arrayOf(processed.buffer),
                        mapOf(1 to encoded.buffer),
                    )
                }
            Log.v("TfliteDetector", "encode($encodeTimeMs ms): ${encoded.buffer.array().toList()}")

            for (i in 0 until decoder.inputTensorCount) {
                val t = decoder.getInputTensor(i)
                Log.v("TfliteDetector", "decoder($i): ${t.name()} / ${t.shape().toList()}")
            }

            // TODO: Loop with encoded into the decoder, accumulating tokens
            val tokenIds = IntBuffer.allocate(1)
            tokenIds.put(2) // start token

            Log.v("TfliteMangaOcr", "output=${decoder.getOutputTensor(0).name()}")
            val logits = decoder.allocateFloatOutputTensor(0)
            val decodeTimeMs =
                measureTimeMillis {
                    decoder.runForMultipleInputsOutputs(
                        arrayOf(encoded.buffer, tokenIds),
                        mapOf(0 to logits.buffer),
                    )
                }
            Log.v("TfliteMangaOcr", "output($decodeTimeMs ms): ${logits.buffer.array().toList()}")

            val maxTokenId = logits.maxValuedIndexInRow(logits.lastRowIndex)
            val token = vocab.lookupToken(maxTokenId)
            if (maxTokenId >= 5) {
//                result.append(token)
                emit(MangaOcr.Result.Partial(token))
            }
            Log.v("TfliteMangaOcr", "Got token $maxTokenId ($token)")

            // TODO:
            emit(MangaOcr.Result.FinalResult(token))
        }.flowOn(Dispatchers.IO)

    companion object {
        data class ModelPath(
            val path: String,
            val sha256: String,
        )

        private val MODEL_ENCODER =
            ModelPath(
                path = "manga-ocr.converted.encoder.tflite",
                sha256 = "",
            )

        private val MODEL_DECODER =
            ModelPath(
                path = "manga-ocr.decoder.tflite",
                sha256 = "",
            )

        suspend fun initialize(context: Context): MangaOcr =
            coroutineScope {
                val encoderFile =
                    async {
                        HfHubRepo("dhleong/manga-ocr-android").resolveLocalPath(
                            context,
                            MODEL_ENCODER.path,
                            sha256 = MODEL_ENCODER.sha256,
                        )
                    }

                val decoderFile =
                    async {
                        HfHubRepo("dhleong/manga-ocr-android").resolveLocalPath(
                            context,
                            MODEL_DECODER.path,
                            sha256 = MODEL_DECODER.sha256,
                        )
                    }
                val vocab = async { Vocab.fetch(context) }
                val initialized =
                    async {
                        TfLite.initialize(context).await()
                    }

                initialized.await()

                val encoder =
                    InterpreterApi.create(
                        encoderFile.await(),
                        InterpreterApi.Options().apply {
                            runtime = InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY
                        },
                    )

                val decoder =
                    InterpreterApi.create(
                        decoderFile.await(),
                        InterpreterApi.Options().apply {
                            runtime = InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY
                        },
                    )

                TfliteMangaOcr(encoder, decoder, vocab.await())
            }
    }
}
