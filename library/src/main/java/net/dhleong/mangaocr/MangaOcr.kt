package net.dhleong.mangaocr

import android.content.Context

interface MangaOcr {
    suspend fun process()

    companion object {
        suspend fun initialize(context: Context): MangaOcr = OrtMangaOcr.initialize(context)
    }
}
