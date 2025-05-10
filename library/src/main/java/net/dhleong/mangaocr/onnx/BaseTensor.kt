package net.dhleong.mangaocr.onnx

open class BaseTensor<T>(
    val buffer: T,
    protected val shape: IntArray,
    val rowsCount: Int,
    val name: String,
) {
    val rowIndices: IntRange
        get() = IntRange(0, (rowsCount - 1).coerceAtLeast(0))

    val colIndices: IntRange
        get() = IntRange(0, (shape[2] - 1).coerceAtLeast(0))

    val lastRowIndex: Int
        get() = rowsCount - 1

    inline fun <T> mapRows(
        quitEarlyOnNull: Boolean = true,
        transform: (row: Int) -> T?,
    ): List<T> {
        val result = mutableListOf<T>()
        for (row in 0 until rowsCount) {
            val transformed = transform(row)
            when {
                transformed == null && quitEarlyOnNull ->
                    return result
                transformed != null ->
                    result.add(transformed)
            }
        }
        return result
    }
}
