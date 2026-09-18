package net.dhleong.mangaocr.onnx

import java.io.Closeable

open class BaseTensor<T>(
    val buffer: T,
    protected val shape: IntArray,
    val rowsCount: Int,
    val name: String,
) : Closeable {
    val rowIndices: IntRange
        get() = IntRange(0, (rowsCount - 1).coerceAtLeast(0))

    val colIndices: IntRange
        get() = IntRange(0, (shape[2] - 1).coerceAtLeast(0))

    val lastRowIndex: Int
        get() = rowsCount - 1

    fun createShapeList() = shape.toList()

    fun createLongShape() = LongArray(shape.size) { shape[it].toLong() }

    inline fun <T> mapRows(
        quitEarlyOnNull: Boolean = true,
        crossinline transform: (row: Int) -> T?,
    ): List<T> = sequenceFromRows(quitEarlyOnNull, transform).toList()

    inline fun <T> sequenceFromRows(
        quitEarlyOnNull: Boolean = true,
        crossinline transform: (row: Int) -> T?,
    ): Sequence<T> =
        sequence {
            for (row in 0 until rowsCount) {
                val transformed = transform(row)
                when {
                    transformed == null && quitEarlyOnNull -> {
                        break
                    }

                    transformed != null -> {
                        yield(transformed)
                    }
                }
            }
        }

    override fun close() {
        // nop by default
    }
}
