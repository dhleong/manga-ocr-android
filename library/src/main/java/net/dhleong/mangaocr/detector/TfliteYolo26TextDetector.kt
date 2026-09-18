package net.dhleong.mangaocr.detector

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import androidx.collection.IntFloatMap
import androidx.collection.intFloatMapOf
import com.google.android.gms.tflite.java.TfLite
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import net.dhleong.mangaocr.Detector
import net.dhleong.mangaocr.ImageProcessor2
import net.dhleong.mangaocr.hub.HfHubRepo
import net.dhleong.mangaocr.hub.ModelPath
import net.dhleong.mangaocr.onnx.FloatTensor.Companion.allocateFloatOutputTensor
import net.dhleong.mangaocr.tflite.ResizeWithPadOp
import net.dhleong.mangaocr.tflite.await
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.support.common.ops.NormalizeOp

/**
 * Old [TfliteMangaTextDetector] detector relied on embedded non-maximum suppression
 * which is no longer supported
 */
class TfliteYolo26TextDetector(
    private val interpreter: InterpreterApi,
    private val model: ModelConfig,
    targetWidth: Int = 640,
    targetHeight: Int = 640,
) : Detector {
    val processor =
        ImageProcessor2(
            ImageProcessor2.OutputFormat.NCHW,
            inputWidth = targetWidth,
            inputHeight = targetHeight,
            bboxFormat = ImageProcessor2.BboxFormat.CenterAndSize,
            resizeOp = ResizeWithPadOp(targetWidth, targetHeight),
            normalizeOp = NormalizeOp(0f, 255f),
        ) { floatBuffer, _ -> floatBuffer }

    override suspend fun process(bitmap: Bitmap): List<Detector.Result> {
        val processed = processor.process(bitmap)
        val outputTensor = interpreter.getOutputTensor(0)

        Log.v(TAG, "output tensors: ${interpreter.outputTensorCount}")
        Log.v(TAG, "output: ${outputTensor.shape().toList()} ${outputTensor.dataType()}")

        val output = interpreter.allocateFloatOutputTensor(0, rowsCountIndex = 2)
        interpreter.run(processed, output.buffer)

        Log.v(TAG, "outputRows=${output.rowsCount}")
        val baseClassIndex = 4

        return output
            .sequenceFromRows(
                quitEarlyOnNull = false,
            ) { i ->
                var maxClass = -1
                var maxConfidence = -1f
                this.model.confidenceThresholds.forEach { classIndex, minConfidence ->
                    val confidenceIndex = baseClassIndex + classIndex
                    val confidence = output[0, confidenceIndex, i]
                    if (confidence >= minConfidence && confidence > maxConfidence) {
                        maxConfidence = confidence
                        maxClass = classIndex
                    }
                }
                if (maxClass >= 0) {
                    val a = output[0, 0, i]
                    val b = output[0, 1, i]
                    val c = output[0, 2, i]
                    val d = output[0, 3, i]
                    val rect = processor.inverseTransform(bitmap, a, b, c, d)
                    Detector.Result(classIndex = maxClass, bbox = Bbox(rect, maxConfidence))
                } else {
                    null
                }
            }.groupBy { it.classIndex }
            .flatMap { (_, boxes) ->
                nonMaximumSuppression(listOf(boxes), threshold = model.nmsThreshold).first()
            }
    }

    data class ModelConfig(
        val path: ModelPath,
        val confidenceThresholds: IntFloatMap,
        val nmsThreshold: Float = DEFAULT_NMS_THRESHOLD,
    )

    companion object {
        private const val TAG = "TfliteYolo26TextDetector"
        private const val DEFAULT_NMS_THRESHOLD = 0.5f

        val YOLO_COCO =
            ModelConfig(
                path =
                    ModelPath(
                        path = "coco-detector-yolos.tflite",
                        sha256 = "066e3a79e587f0ed4de0dfcd7c0d2d8cad856ac7eafdafb92733166f46b0f31c",
                    ),
                confidenceThresholds =
                    intFloatMapOf(
                        0,
                        0.15f, // text
                        1,
                        0.45f, // onomatopoeia
                    ),
            )

        val YOLO_COCO_QUANTIZED =
            YOLO_COCO.copy(
                path =
                    ModelPath(
                        path = "coco-detector-yolos-w8a32.tflite",
                        sha256 = "38ee43a8bafc7ab1148b0c9ab5a52b4f4ccc0727245240ddb23f2f9943123aa0",
                    ),
            )

        suspend fun initialize(
            context: Context,
            model: ModelConfig = YOLO_COCO_QUANTIZED,
        ): Detector =
            coroutineScope {
                val modelFile =
                    async {
                        HfHubRepo("dhleong/manga-ocr-android")
                            .resolveLocalPath(context, model.path)
                    }

                val initialized = async { TfLite.initialize(context).await() }

                initialized.await()
                val interpreter =
                    InterpreterApi.create(
                        modelFile.await(),
                        InterpreterApi.Options().apply {
                            runtime = InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY
                        },
                    )

                TfliteYolo26TextDetector(
                    interpreter,
                    model = model,
                )
            }
    }
}
