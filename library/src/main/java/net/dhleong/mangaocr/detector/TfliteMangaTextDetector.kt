package net.dhleong.mangaocr.detector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import com.google.android.gms.tflite.java.TfLite
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import net.dhleong.mangaocr.Detector
import net.dhleong.mangaocr.hub.HfHubRepo
import net.dhleong.mangaocr.onnx.FloatTensor.Companion.allocateFloatOutputTensor
import net.dhleong.mangaocr.tflite.await
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

class TfliteMangaTextDetector(
//    private val objectDetector: ObjectDetector,
    private val interpreter: InterpreterApi,
) : Detector {
//    override suspend fun process(bitmap: Bitmap): List<Detector.Result> {
//        val imageProcessor =
//            ImageProcessor
//                .Builder()
//                .add(ResizeOp(640, 640, ResizeOp.ResizeMethod.BILINEAR))
//                .build()
//        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(bitmap))
//        val results = objectDetector.detect(tensorImage)
//        return results.map { result ->
//            Log.v("TfliteDetector", "found ${result.categories[0].label} ${result.categories[0].score}")
//            Detector.Result(
//                classIndex = result.categories[0].index,
//                bbox = Bbox(result.boundingBox, result.categories[0].score),
//            )
//        }
//    }

    override suspend fun process(bitmap: Bitmap): List<Detector.Result> {
        val targetWidth = 640
        val targetHeight = 640
        val imageProcessor =
            ImageProcessor
                .Builder()
                .add(ResizeOp(targetHeight, targetWidth, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0f, 255f)) // [ 0, 1 ]
//                .add(NormalizeOp(127.5f, 127.5f)) // [ -0.5, 0.5 ]
                .build()
        val processed = imageProcessor.process(TensorImage.fromBitmap(bitmap))

        // NOTE: The coord *appear* to be a percentage of the input dimension, so:
        //   coordPercent * inputWidth * (originalWidth / inputWidth)
        // cancels out to `coordPercent * originalWidth` to get the "real" dimens.
        // Unfortunately getting garbage output using this buuuut...
        val widthRatio = bitmap.width
        val heightRatio = bitmap.height

        val outputTensor = interpreter.getOutputTensor(0)
        Log.v("TfliteDetector", "output: ${outputTensor.shape().toList()} ${outputTensor.dataType()}")

        val output = interpreter.allocateFloatOutputTensor(0)
        interpreter.run(processed.buffer, output.buffer)
        Log.v("TfliteDetector", "output: ${output.buffer.array().toList()}")

        return output.indices.mapNotNull { i ->
            val confidence = output[0, i, 4]
            if (confidence < CONFIDENCE_THRESHOLD) {
                return@mapNotNull null
            }

            val centerX = output[0, i, 0] * widthRatio
            val centerY = output[0, i, 1] * heightRatio
            val width = output[0, i, 2] * widthRatio
            val height = output[0, i, 3] * heightRatio

            Detector.Result(
                bbox = Bbox(RectF(centerX - width / 2, centerY - height / 2, centerX + width / 2, centerY + height / 2), confidence),
                classIndex = output[0, i, 5].toInt(),
            )
        }
    }

    companion object {
        private const val CONFIDENCE_THRESHOLD = 0.5f

        data class ModelPath(
            val path: String,
            val sha256: String,
        )

        private val MODEL_FLOAT32 =
            ModelPath(
                path = "manga-text-detector_float32.tflite",
                sha256 = "cc8f6894424cebaeda7b3950004b77838967ba28e6d3b01bf94025689eebde40",
            )

        suspend fun initialize(
            context: Context,
            model: ModelPath = MODEL_FLOAT32,
        ): Detector =
            coroutineScope {
                val modelFile =
                    async {
                        HfHubRepo("dhleong/manga-ocr-android").resolveLocalPath(
                            context,
                            model.path,
                            sha256 = model.sha256,
                        )
                    }

                val initialized =
                    async {
                        TfLite.initialize(context).await()
                    }

//                val options =
//                    ObjectDetector.ObjectDetectorOptions.builder().apply {
//                        setScoreThreshold(0.5f)
//                        setBaseOptions(
//                            BaseOptions
//                                .builder()
//                                .apply {
//                                    setNumThreads(4)
//                                    useNnapi()
//                                }.build(),
//                        )
//                    }
//
//                val detector = ObjectDetector.createFromFileAndOptions(modelFile, options.build())
//
//                TfliteMangaTextDetector(detector)
                // NOTE: ObjectDetector complains about needing metadata. See here:
                // https://ai.google.dev/edge/litert/models/metadata_writer_tutorial#object_detectors
                // Sadly, tflite-support won't install on my mac :(

                initialized.await()
                val interpreter =
                    InterpreterApi.create(
                        modelFile.await(),
                        InterpreterApi.Options().apply {
                            runtime = InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY
                        },
                    )

                TfliteMangaTextDetector(interpreter)
            }
    }
}
