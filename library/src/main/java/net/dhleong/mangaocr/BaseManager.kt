package net.dhleong.mangaocr

import android.content.Context
import androidx.lifecycle.AtomicReference
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleCoroutineScope
import androidx.lifecycle.repeatOnLifecycle
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch

abstract class BaseManager<T>(
    context: Context,
    scope: LifecycleCoroutineScope,
    private val lifecycle: Lifecycle,
) {
    private val model: AtomicReference<T?> = AtomicReference(null)
    private val modelLoaded = MutableSharedFlow<Unit>()

    init {
        scope.launch {
            lifecycle.repeatOnLifecycle(Lifecycle.State.RESUMED) {
                if (model.get() == null) {
                    model.set(initialize(context))
                    modelLoaded.emit(Unit)
                }
            }
        }
    }

    protected abstract suspend fun initialize(context: Context): T

    protected suspend fun awaitModel(): T {
        val existing = model.get()
        if (existing != null) {
            return existing
        }

        modelLoaded.first()
        return requireNotNull(model.get())
    }
}
