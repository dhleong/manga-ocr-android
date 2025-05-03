package net.dhleong.mangaocr.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxValue
import org.tensorflow.lite.InterpreterApi
import java.nio.FloatBuffer

class FloatTensor(
    val buffer: FloatBuffer,
    private val shape: IntArray,
    val rowsCount: Int,
) {
    val indices: IntRange
        get() = IntRange(0, (rowsCount - 1).coerceAtLeast(0))

    operator fun get(
        x: Int,
        y: Int,
        z: Int,
    ): Float {
        val i =
            x * shape[1] * shape[2] +
                y * shape[2] +
                z
        return buffer.get(i)
    }

    companion object {
        fun from(
            tensor: OnnxValue,
            rowsCountIndex: Int = 1,
        ): FloatTensor {
            val onnxTensor = tensor as OnnxTensor
            return FloatTensor(
                requireNotNull(onnxTensor.floatBuffer),
                IntArray(onnxTensor.info.shape.size) { i ->
                    onnxTensor.info.shape[i].toInt()
                },
                rowsCount = onnxTensor.info.shape[rowsCountIndex].toInt(),
            )
        }

        fun InterpreterApi.allocateFloatOutputTensor(
            outputIndex: Int = 0,
            rowsCountIndex: Int = 1,
        ): FloatTensor {
            val tensor = getOutputTensor(outputIndex)
            val shape = tensor.shape()
            val buffer = FloatBuffer.allocate(shape.reduce { acc, i -> acc * i })
            return FloatTensor(
                buffer,
                shape,
                rowsCount = shape[rowsCountIndex],
            )
        }
    }
}
