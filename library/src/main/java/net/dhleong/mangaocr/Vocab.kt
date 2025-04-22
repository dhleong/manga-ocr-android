package net.dhleong.mangaocr

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

class Vocab private constructor(
    private val dict: List<String>,
) {
    fun lookupToken(tokenId: Int) = dict[tokenId]

    override fun toString(): String = "Vocab(${dict.size})"

    companion object {
        suspend fun loadFromFile(file: File): Vocab =
            withContext(Dispatchers.IO) {
                Vocab(file.readLines())
            }
    }
}
