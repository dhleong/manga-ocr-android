package net.dhleong.mangaocr.onnx

import org.tensorflow.lite.InterpreterApi
import java.nio.IntBuffer

class IntTensor(
    buffer: IntBuffer,
    shape: IntArray,
    rowsCount: Int,
    name: String,
) : BaseTensor<IntBuffer>(buffer, shape, rowsCount, name) {
    operator fun get(
        x: Int,
        y: Int,
        z: Int,
    ): Int {
        val i =
            x * shape[1] * shape[2] +
                y * shape[2] +
                z
        return buffer.get(i)
    }

    companion object {
        fun InterpreterApi.allocateIntOutputTensor(
            outputIndex: Int = 0,
            rowsCountIndex: Int = 1,
        ): IntTensor {
            val tensor = getOutputTensor(outputIndex)
            val shape = tensor.shape()
            val buffer = IntBuffer.allocate(shape.reduce { acc, i -> acc * i })
            return IntTensor(
                buffer,
                shape,
                rowsCount = shape[rowsCountIndex],
                name = tensor.name(),
            )
        }
    }
}
