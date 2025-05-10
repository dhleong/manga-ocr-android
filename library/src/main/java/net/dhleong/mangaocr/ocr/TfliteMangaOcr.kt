package net.dhleong.mangaocr.ocr

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
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
import net.dhleong.mangaocr.onnx.createSession
import java.io.File
import java.nio.LongBuffer

class TfliteMangaOcr(
//    private val encoder: InterpreterApi,
//    private val decoder: InterpreterApi,
    private val encoder: OrtSession,
    private val decoder: OrtSession,
    private val vocab: Vocab,
    private val maxChars: Int = 300,
    private val targetWidth: Int = 224,
    private val targetHeight: Int = 224,
) : MangaOcr {
    override suspend fun process(bitmap: Bitmap): Flow<MangaOcr.Result> =
        flow {
            val encoded = encodeOnnx(bitmap)

            val tokenIds = LongBuffer.allocate(maxChars)
            tokenIds.put(2) // start token
            val result = StringBuilder()

            for (tokensCount in 1 until maxChars) {
                // Prepare to read the tokenIds:
                tokenIds.flip()
                val logits = decodeOnnx(encoded, tokenIds, tokensCount)

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
                Log.v("TfliteMangaOcr", "Got token $maxTokenId ($token)")
            }

            emit(MangaOcr.Result.FinalResult(result.toString()))
        }.flowOn(Dispatchers.IO)

//    private fun encode(bitmap: Bitmap): FloatTensor {
//        val imageProcessor =
//            ImageProcessor
//                .Builder()
//                .add(ResizeOp(targetHeight, targetWidth, ResizeOp.ResizeMethod.BILINEAR))
// //                .add(NormalizeOp(0f, 255f)) // [ 0, 1 ]
//                .add(NormalizeOp(127.5f, 127.5f)) // [ -1, 1 ]
//                .build()
//        val processed = imageProcessor.process(TensorImage.fromBitmap(bitmap))
//
//        for (i in 0 until encoder.outputTensorCount) {
//            val t = encoder.getOutputTensor(i)
//            Log.v("TfliteDetector", "encoder($i): ${t.name()} / ${t.shape().toList()}")
//        }
//
//        val encoded = encoder.allocateFloatOutputTensor(1)
//        Log.v("TfliteDetector", "encode(...): ${encoded.name}=${encoded.buffer.array().toList()}")
//        val encodeTimeMs =
//            measureTimeMillis {
//                encoder.runForMultipleInputsOutputs(
//                    arrayOf(processed.buffer),
//                    mapOf(1 to encoded.buffer),
//                )
//            }
//        Log.v("TfliteDetector", "encode($encodeTimeMs ms): ${encoded.buffer.array().toList()}")
//        return encoded
//    }

    private fun encodeOnnx(bitmap: Bitmap): FloatTensor {
        val imageProcessor =
            net.dhleong.mangaocr.ImageProcessor { floats, shape ->
                OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), floats, shape)
            }

        val processed = imageProcessor.preprocess(bitmap)

//        val encoded = encoder.allocateFloatOutputTensor(1)
//        Log.v("TfliteDetector", "encode(...): ${encoded.name}=${encoded.buffer.array().toList()}")
        val start = System.currentTimeMillis()
        val outputs = encoder.run(mapOf("pixel_values" to processed))
        val hiddenStatesTensor = FloatTensor.from(outputs.get("encoder_hidden_states").get())
        Log.v("TfliteDetector", "encode.outputs: ${outputs.toList()}")
        if (hiddenStatesTensor.rowsCount == 0) {
            throw IllegalStateException("No resultRows")
        }

        val encodeTimeMs = System.currentTimeMillis() - start
        val encoded = hiddenStatesTensor
        Log.v("TfliteDetector", "encode($encodeTimeMs ms -> ${encoded.createShapeList()}): ${encoded.buffer.array().toList()}")
        return encoded
    }

    private fun decodeOnnx(
        encoded: FloatTensor,
        tokenIds: LongBuffer,
        tokensCount: Int,
    ): FloatTensor {
        encoded.buffer.limit(encoded.buffer.capacity()) // FIXME ?
//        encoded.buffer.limit(768) // FIXME
        encoded.buffer.position(0)

        val tokenIdsTensor =
            OnnxTensor.createTensor(
                OrtEnvironment.getEnvironment(),
                tokenIds,
                longArrayOf(1, tokensCount.toLong()),
            )
        val hiddenStatesTensor =
            OnnxTensor.createTensor(
                OrtEnvironment.getEnvironment(),
                encoded.buffer,
                encoded.createLongShape(),
//                longArrayOf(1, 197, 768), // FIXME
            )
        Log.v("TfliteMangaOcr", "inputInfo=${decoder.inputInfo} ${decoder.inputInfo["encoder_hidden_states"]?.info}")
        val start = System.currentTimeMillis()
        val outputs = decoder.run(mapOf("encoder_hidden_states" to hiddenStatesTensor, "input_ids" to tokenIdsTensor))
        val logitsTensor = FloatTensor.from(outputs.get("logits").get())
        if (logitsTensor.rowsCount == 0) {
            throw IllegalStateException("No resultRows")
        }

        val delta = System.currentTimeMillis() - start
        Log.v("TFLiteMangaOcr", "logits($delta ms; ${logitsTensor.rowsCount} rows): ${logitsTensor.createShapeList()}")
        return logitsTensor
    }

//    private fun decodeTflite(encoded: FloatTensor): FloatTensor {
//        for (i in 0 until decoder.inputTensorCount) {
//            val t = decoder.getInputTensor(i)
//            Log.v("TfliteDetector", "decoder($i): ${t.name()} / ${t.shape().toList()}")
//        }
//
//        // TODO: Loop with encoded into the decoder, accumulating tokens
//        val tokenIds = IntBuffer.allocate(1)
//        tokenIds.put(2) // start token
//
//        Log.v("TfliteMangaOcr", "output=${decoder.getOutputTensor(0).name()}")
//        val logits = decoder.allocateFloatOutputTensor(0)
//        val decodeTimeMs =
//            measureTimeMillis {
//                decoder.runForMultipleInputsOutputs(
//                    arrayOf(encoded.buffer, tokenIds),
//                    mapOf(0 to logits.buffer),
//                )
//            }
//        Log.v("TfliteMangaOcr", "output($decodeTimeMs ms): ${logits.buffer.array().toList()}")
//        return logits
//    }

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

        private val ONNX_MODEL_ENCODER =
            ModelPath(
                path = "manga-ocr.converted.encoder.onnx",
                sha256 = "",
            )

        private val ONNX_MODEL_DECODER =
            ModelPath(
                path = "manga-ocr.converted.decoder.onnx",
                sha256 = "",
            )

        suspend fun initialize(context: Context): MangaOcr =
            coroutineScope {
                val repo = HfHubRepo("dhleong/manga-ocr-android")
                val encoderFile = async { repo.resolveLocalPath(context, ONNX_MODEL_ENCODER) }
                val decoderFile = async { repo.resolveLocalPath(context, ONNX_MODEL_DECODER) }
                val vocab = async { Vocab.fetch(context) }
//                val initialized = async { TfLite.initialize(context).await() }

//                initialized.await()

//                val encoder =
//                    InterpreterApi.create(
//                        encoderFile.await(),
//                        InterpreterApi.Options().apply {
//                            runtime = InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY
//                        },
//                    )

//                val decoder =
//                    InterpreterApi.create(
//                        decoderFile.await(),
//                        InterpreterApi.Options().apply {
//                            runtime = InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY
//                        },
//                    )

                val encoder = buildSession(encoderFile.await())
                val decoder = buildSession(decoderFile.await())

                TfliteMangaOcr(encoder, decoder, vocab.await())
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
