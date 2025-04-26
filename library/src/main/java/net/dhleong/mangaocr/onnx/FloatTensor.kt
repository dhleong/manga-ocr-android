package net.dhleong.mangaocr.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxValue

class FloatTensor(
    tensor: OnnxTensor,
) {
    private val buffer = requireNotNull(tensor.floatBuffer)
    val shape = tensor.info.shape

    operator fun get(
        x: Int,
        y: Int,
        z: Int,
    ): Float {
        val i =
            x * shape[1].toInt() * shape[2].toInt() +
                y * shape[2].toInt() +
                z
        return buffer.get(i)
    }

    companion object {
        fun from(tensor: OnnxValue) = FloatTensor(tensor as OnnxTensor)
    }
}
