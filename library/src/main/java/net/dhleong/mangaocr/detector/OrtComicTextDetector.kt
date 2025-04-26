package net.dhleong.mangaocr.detector

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxTensorLike
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import net.dhleong.mangaocr.Detector
import net.dhleong.mangaocr.ImageProcessor
import net.dhleong.mangaocr.hub.HfHubRepo
import net.dhleong.mangaocr.onnx.FloatTensor
import net.dhleong.mangaocr.onnx.createSession
import java.io.File

class OrtComicTextDetector private constructor(
    private val session: OrtSession,
    // NOTE: Currently this drops boxes unexpectedly
    private val performNonMaxSuppression: Boolean = false,
    private val processor: ImageProcessor<OnnxTensorLike> =
        ImageProcessor(
            inputWidth = 1024,
            inputHeight = 1024,
            normalize = { it }, // no-op
            grayscaleify = false,
        ) { floats, shape ->
            OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), floats, shape)
        },
) : Detector {
    override suspend fun process(bitmap: Bitmap): List<Detector.Result> =
        withContext(Dispatchers.IO) {
            val image = processor.preprocess(bitmap)
            val widthRatio = bitmap.width / image.info.shape[2].toFloat()
            val heightRatio = bitmap.height / image.info.shape[3].toFloat()

            val outputs = session.run(mapOf("images" to image))
            val blocks = FloatTensor.from(outputs.get("blk").get())

            val boxes: List<MutableList<Bbox>> = listOf(mutableListOf(), mutableListOf())
            for (i in 0 until blocks.shape[1].toInt()) {
                val confidence = blocks[0, i, 4]
                if (confidence < CONFIDENCE_THRESHOLD) {
                    continue
                }

                var classIndex = 0
                if (blocks[0, i, 5] < blocks[0, i, 6]) {
                    classIndex = 1
                }

                val centerX = blocks[0, i, 0] * widthRatio
                val centerY = blocks[0, i, 1] * heightRatio
                val width = blocks[0, i, 2] * widthRatio
                val height = blocks[0, i, 3] * heightRatio

                val rect =
                    RectF(
                        centerX - width / 2,
                        centerY - height / 2,
                        centerX + width / 2,
                        centerY + height / 2,
                    )

                boxes[classIndex] += Bbox(rect, confidence)
            }

            val filtered =
                if (performNonMaxSuppression) {
                    nonMaximumSuppression(boxes, threshold = NMS_THRESHOLD)
                } else {
                    boxes
                }
            filtered.flatMapIndexed { i, classBoxes ->
                classBoxes.asSequence().map { Detector.Result(it, classIndex = i) }
            }
        }

    companion object {
        private const val CONFIDENCE_THRESHOLD = 0.5f
        private const val NMS_THRESHOLD = 0.5f

        suspend fun initialize(context: Context): Detector {
            val start = System.currentTimeMillis()
            val ctx = context.applicationContext
            val modelPath =
                HfHubRepo("dhleong/manga-ocr-android").resolveLocalPath(
                    ctx,
                    "comictextdetector.preprocessed.quant.onnx",
                    sha256 = "7378ce5a6b01c1953d794b404ec1ce92f7a0f71cd9ed15eec753a61e1707d8e5",
                )
            Log.v("OrtComicTextDetector", "Prepared model file in ${System.currentTimeMillis() - start} ms")
            return OrtComicTextDetector(buildSession(modelPath))
        }

        private suspend fun buildSession(modelPath: File) =
            createSession(modelPath) {
                setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
    }
}
