package net.dhleong.mangaocr.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxValue
import org.tensorflow.lite.InterpreterApi
import java.nio.FloatBuffer

class FloatTensor(
    buffer: FloatBuffer,
    shape: IntArray,
    rowsCount: Int,
    name: String,
) : BaseTensor<FloatBuffer>(buffer, shape, rowsCount, name) {
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

    fun maxValuedIndexInRow(row: Int): Int {
        var maxTokenId = -1
        var maxArg = 0f
        for (i in colIndices) {
            // find argmax
            val value = this[0, row, i]
            if (maxTokenId < 0 || value > maxArg) {
                maxTokenId = i
                maxArg = value
            }
        }

        if (maxTokenId < 0) {
            throw IllegalStateException("No max value found")
        }

        return maxTokenId
    }

    fun assertHasRows(): FloatTensor {
        if (rowsCount == 0) {
            throw IllegalStateException("No resultRows")
        }
        return this
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
                name = onnxTensor.info.dimensionNames.joinToString(","),
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
                name = tensor.name(),
            )
        }
    }
}
