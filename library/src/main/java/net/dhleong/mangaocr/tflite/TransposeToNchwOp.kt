package net.dhleong.mangaocr.tflite

import android.graphics.PointF
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ColorSpaceType
import org.tensorflow.lite.support.image.ImageOperator
import org.tensorflow.lite.support.image.ImageProperties
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder

class TransposeToNchwOp : ImageOperator {
    override fun apply(image: TensorImage): TensorImage {
        val letterboxed = image.bitmap

        val pixels = IntArray(image.width * image.height)
        letterboxed.getPixels(pixels, 0, image.width, 0, 0, image.width, image.height)

        val byteBuffer =
            ByteBuffer
                .allocateDirect(1 * 3 * image.width * image.height * 4)
                .order(ByteOrder.nativeOrder())
        val floatBuffer = byteBuffer.asFloatBuffer()

        val area = image.width * image.height
        for (i in pixels.indices) {
            val pixel = pixels[i]
            floatBuffer.put(i, ((pixel shr 16) and 0xFF).toFloat())
            floatBuffer.put(area + i, ((pixel shr 8) and 0xFF).toFloat())
            floatBuffer.put(2 * area + i, (pixel and 0xFF).toFloat())
        }
        byteBuffer.rewind()
        return TensorImage(DataType.FLOAT32).apply {
            load(
                TensorBuffer.createDynamic(DataType.FLOAT32).apply {
                    // NOTE: This is a lie: it's actually 1, 3, h, w...
                    // but tflite barfs if we tell the truth :(
                    loadBuffer(byteBuffer, intArrayOf(1, image.width, image.height, 3))
                },
            )
        }
    }

    override fun getOutputImageWidth(
        inputImageHeight: Int,
        inputImageWidth: Int,
    ): Int = inputImageWidth

    override fun getOutputImageHeight(
        inputImageHeight: Int,
        inputImageWidth: Int,
    ): Int = inputImageHeight

    override fun inverseTransform(
        point: PointF?,
        inputImageHeight: Int,
        inputImageWidth: Int,
    ): PointF? = point
}
