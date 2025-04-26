package net.dhleong.mangaocr

// ImageProcessor.kt
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.ColorSpace
import android.os.Build
import android.util.Log
import androidx.core.graphics.get
import androidx.core.graphics.scale
import androidx.core.graphics.toColorInt
import java.nio.FloatBuffer

class ImageProcessor<T>(
    private val inputHeight: Int = 224,
    private val inputWidth: Int = 224,
    private val normalize: (Float) -> Float = { (it - 0.5f) / 0.5f },
    private val grayscaleify: Boolean = GRAYSCALEIFY,
    private val floatsToTensor: (FloatBuffer, shape: LongArray) -> T,
) {
    // Allocate buffer for tensor
//    private val buffer = FloatBuffer.allocate(3 * inputHeight * inputWidth)
    private val shape = longArrayOf(1, 3, inputHeight.toLong(), inputWidth.toLong())

    fun preprocess(bitmap: Bitmap): T {
        val start = System.currentTimeMillis()

        val resizedBitmap = bitmap.resizeTo(inputWidth, inputHeight)

        Log.v("ImageProcessor", "resized ${bitmap.width} / ${bitmap.height} to ${resizedBitmap.width} / ${resizedBitmap.height}")
        val buffer = FloatBuffer.allocate(3 * inputHeight * inputWidth)
        buffer.clear()
        buffer.limit(buffer.capacity())

        // Convert bitmap to normalized float tensor
        // The shape here seems to be:
        // R[height, width], G[height, width], B[height, width]
        val w = resizedBitmap.width
        val h = resizedBitmap.height
        for (y in 0 until h) {
            for (x in 0 until w) {
                var pixel = resizedBitmap[x, y]

//                // NOTE: This *works* but seems less accurate than the original colors?
//                // Translate [0, 1] -> [0, 255]
//                val gray = Color.luminance(pixel) * 255f
                if (grayscaleify) {
//                    val gray = (Color.luminance(pixel) * 255f).toInt()
//                    pixel = Color.argb(255, gray, gray, gray)

                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                        pixel =
                            Color
                                .convert(
                                    pixel,
                                    ColorSpace.get(ColorSpace.Named.BT2020_PQ),
                                ).toColorInt()
                    }
                }
//                val r = gray
//                val g = gray
//                val b = gray

                val r = Color.red(pixel)
                val g = Color.green(pixel)
                val b = Color.blue(pixel)

                val ri = y * inputWidth + x
                val gi = 1 * inputWidth * inputHeight + ri
                val bi = 2 * inputWidth * inputHeight + ri
                buffer.put(ri, normalize(r / 255f))
                buffer.put(gi, normalize(g / 255f))
                buffer.put(bi, normalize(b / 255f))
            }
        }

        if (resizedBitmap !== bitmap) {
            resizedBitmap.recycle()
        }

        val tensor = floatsToTensor(buffer, shape)
        Log.v("ImageProcessor", "preprocessed ($inputWidth x $inputHeight) in ${System.currentTimeMillis() - start}ms")
        return tensor
    }

    companion object {
        // Koharu seems to preserve it, but for whatever reason... that doesn't work so well for us
        private const val PRESERVE_ASPECT = false
        private const val GRAYSCALEIFY = true

        fun Bitmap.resizeTo(
            inputWidth: Int,
            inputHeight: Int,
        ): Bitmap {
            val resizedBitmap =
                if (PRESERVE_ASPECT) {
                    val scale: Float =
                        if (width > height) {
                            inputWidth.toFloat() / width
                        } else {
                            inputHeight.toFloat() / height
                        }

                    val newWidth = (width * scale).toInt()
                    val newHeight = (height * scale).toInt()
                    scale(newWidth, newHeight)
                } else {
                    scale(inputWidth, inputHeight)
                }

            return resizedBitmap
        }
    }
}
