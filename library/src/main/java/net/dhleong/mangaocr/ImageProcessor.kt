package net.dhleong.mangaocr

// ImageProcessor.kt
import android.graphics.Bitmap
import android.graphics.Color
import java.nio.FloatBuffer

class ImageProcessor<T>(
    private val floatsToTensor: (FloatBuffer, shape: LongArray) -> T,
) {
    // Size expected by the ViT model
    private val inputHeight = 224
    private val inputWidth = 224

    // Normalization parameters from ViTImageProcessor
    private val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val std = floatArrayOf(0.229f, 0.224f, 0.225f)

    // Allocate buffer for tensor
    private val buffer = FloatBuffer.allocate(3 * inputHeight * inputWidth)
    private val shape = longArrayOf(1, 3, inputHeight.toLong(), inputWidth.toLong())

    fun preprocess(bitmap: Bitmap): T {
        // Convert to RGB if needed
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)
        buffer.clear()
        buffer.limit(buffer.capacity())

        // Convert bitmap to normalized float tensor
        for (y in 0 until inputHeight) {
            for (x in 0 until inputWidth) {
                val pixel = resizedBitmap.getPixel(x, y)

                // in [0, 1]
                val gray = Color.luminance(pixel)

                // normalize to [-1, 1]
                val normalized = (gray - 0.5f) / 0.5f

//                // Extract RGB values
//                val r = Color.red(pixel) / 255.0f
//                val g = Color.green(pixel) / 255.0f
//                val b = Color.blue(pixel) / 255.0f
//
//                // Normalize and store in CHW order (PyTorch format)
//                buffer.put((r - mean[0]) / std[0])
//                buffer.put((g - mean[1]) / std[1])
//                buffer.put((b - mean[2]) / std[2])

                val j = x * y
                buffer.put(j, normalized)
                buffer.put(j + inputHeight * inputWidth, normalized)
                buffer.put(j + 2 * inputHeight * inputWidth, normalized)
//                buffer.put(normalized)
//                buffer.put(normalized)
//                buffer.put(normalized)
            }
        }

//        buffer.flip()

        if (resizedBitmap !== bitmap) {
            resizedBitmap.recycle()
        }

        // Create tensor with shape [1, 3, 224, 224]
        return floatsToTensor(buffer, shape)
    }
}
