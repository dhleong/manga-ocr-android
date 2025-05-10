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
import kotlin.jvm.optionals.getOrNull

class OrtComicTextDetector private constructor(
    private val session: OrtSession,
    inputWidth: Int = 1024,
    inputHeight: Int = 1024,
    private val processor: ImageProcessor<OnnxTensorLike> =
        ImageProcessor(
            inputWidth = inputWidth,
            inputHeight = inputHeight,
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

            Log.v("ORT", "${session.inputNames} -> ${session.outputNames}")
            val outputs = session.run(mapOf("images" to image))
            val output =
                outputs.get("blk").getOrNull()
                    ?: outputs.get("output0").get()
            val blocks = FloatTensor.from(output)

            val boxes: List<MutableList<Bbox>> = listOf(mutableListOf(), mutableListOf())
            for (i in blocks.rowIndices) {
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

            nonMaximumSuppression(boxes, threshold = NMS_THRESHOLD)
                .flatMapIndexed { i, classBoxes ->
                    classBoxes.asSequence().map { Detector.Result(it, classIndex = i) }
                }.also { flattened ->
                    Log.v("OrtComicTextDetector", "Filtered ${boxes.sumOf { it.size }} -> ${flattened.size} boxes")
                }
        }

    companion object {
        private const val CONFIDENCE_THRESHOLD = 0.5f
        private const val NMS_THRESHOLD = 0.5f

        private const val USE_MANGA_DETECTOR = false

        suspend fun initialize(context: Context): Detector {
            val ctx = context.applicationContext
            return if (USE_MANGA_DETECTOR) {
                // TODO: We may need some different approach to extracting Bboxes
                initializeMangaTextDetector(ctx)
            } else {
                initializeComicTextDetector(ctx)
            }
        }

        private suspend fun initializeComicTextDetector(context: Context) =
            initializeFromPath {
                HfHubRepo("dhleong/manga-ocr-android").resolveLocalPath(
                    context,
                    "comictextdetector.preprocessed.quant.onnx",
                    sha256 = "7378ce5a6b01c1953d794b404ec1ce92f7a0f71cd9ed15eec753a61e1707d8e5",
                )
            }

        private suspend fun initializeMangaTextDetector(context: Context) =
            initializeFromPath(inputSize = 640) {
                HfHubRepo("dhleong/manga-ocr-android").resolveLocalPath(
                    context,
                    "manga-text-detector.onnx",
                    sha256 = "92403c26623702d0938acda6bef5e3dff245529499e8610a5d6c3cea5b0b2455",
                )
            }

        private suspend fun initializeFromPath(
            inputSize: Int = 1024,
            resolvePath: suspend () -> File,
        ): Detector {
            val start = System.currentTimeMillis()
            val modelPath = resolvePath()
            Log.v("OrtComicTextDetector", "Prepared model file in ${System.currentTimeMillis() - start} ms")
            return OrtComicTextDetector(buildSession(modelPath), inputWidth = inputSize, inputHeight = inputSize)
        }

        private suspend fun buildSession(modelPath: File) =
            createSession(modelPath) {
                setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
    }
}
