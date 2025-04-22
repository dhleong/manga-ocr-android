package net.dhleong.mangaocr

import android.content.Context
import androidx.lifecycle.AtomicReference
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleCoroutineScope
import androidx.lifecycle.repeatOnLifecycle
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch

class MangaOcrManager(
    context: Context,
    private val scope: LifecycleCoroutineScope,
    private val lifecycle: Lifecycle,
) {
    private val model: AtomicReference<MangaOcr?> = AtomicReference(null)
    private val modelLoaded = MutableSharedFlow<Unit>()

    init {
        scope.launch {
            lifecycle.repeatOnLifecycle(Lifecycle.State.RESUMED) {
                if (model.get() == null) {
                    model.set(MangaOcr.initialize(context))
                    modelLoaded.emit(Unit)
                }
            }
        }
    }

    private suspend fun awaitModel(): MangaOcr {
        val existing = model.get()
        if (existing != null) {
            return existing
        }

        modelLoaded.first()
        return requireNotNull(model.get())
    }

    suspend fun process() {
        awaitModel().process()
    }
}
