package net.dhleong.mangaocr

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import net.dhleong.mangaocr.hub.HfHubRepo
import java.io.File

class Vocab private constructor(
    private val dict: List<String>,
) {
    val size: Int get() = dict.size

    fun lookupToken(tokenId: Int) = dict[tokenId]

    override fun toString(): String = "Vocab(${dict.size})"

    companion object {
        suspend fun loadFromFile(file: File): Vocab =
            withContext(Dispatchers.IO) {
                Vocab(file.readLines())
            }

        suspend fun load(context: Context): Vocab {
            val start = System.currentTimeMillis()
            val vocabPath =
                HfHubRepo("mayocream/koharu").resolveLocalPath(
                    context,
                    "vocab.txt",
                    sha256 = "344fbb6b8bf18c57839e924e2c9365434697e0227fac00b88bb4899b78aa594d",
                )
            Log.v("Tflite", "Prepared vocab file in ${System.currentTimeMillis() - start} ms")
            return loadFromFile(vocabPath)
        }
    }
}
