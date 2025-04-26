package net.dhleong.mangaocr.detector

import kotlin.math.max
import kotlin.math.min

fun nonMaximumSuppression(
    boxes: List<List<Bbox>>,
    threshold: Float,
): List<List<Bbox>> {
    // borrowed from candle-transformers Rust module
    return boxes.map { boxesForClass ->
        val sorted = boxesForClass.toMutableList()
        sorted.sortByDescending { it.confidence }

        var currentIndex = 0
        for (i in sorted.indices) {
            val drop =
                (0 until currentIndex).any { prevIndex ->
                    val iou = iou(sorted[prevIndex], sorted[i])
                    iou > threshold
                }
            if (!drop) {
                val v = sorted[currentIndex]
                sorted[currentIndex] = sorted[i]
                sorted[i] = v
                ++currentIndex
            }
        }

        sorted.take(currentIndex)
    }
}

/** Intersection over union of two bounding boxes. */
private fun iou(
    a: Bbox,
    b: Bbox,
): Float {
    val aArea = (a.rect.width() + 1f) * (a.rect.height() + 1f)
    val bArea = (b.rect.width() + 1f) * (b.rect.height() + 1f)
    val xmin = max(a.rect.left, b.rect.left)
    val xmax = min(a.rect.right, b.rect.right)
    val ymin = max(a.rect.top, b.rect.top)
    val ymax = min(a.rect.bottom, b.rect.bottom)
    val iArea = max(0f, (xmax - xmin + 1f) * (ymax - ymin + 1f))
    return iArea / (aArea + bArea - iArea)
}
