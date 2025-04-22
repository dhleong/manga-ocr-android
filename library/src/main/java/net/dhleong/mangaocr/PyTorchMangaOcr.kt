package net.dhleong.mangaocr

import android.content.Context
import android.graphics.Bitmap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.pytorch.executorch.Module

class PyTorchMangaOcr private constructor(
    private val module: Module,
) : MangaOcr {
    override suspend fun process(bitmap: Bitmap) {
        module.execute("run")
        // TODO:
    }

    companion object {
        suspend fun initialize(context: Context): PyTorchMangaOcr {
            // TODO: Check cached model
            val module =
                withContext(Dispatchers.IO) {
                    Module.load("path", Module.LOAD_MODE_MMAP)
                }
            return PyTorchMangaOcr(module)
        }
    }
}
