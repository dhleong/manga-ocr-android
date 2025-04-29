package net.dhleong.mangaocr.detector

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
    val xA = maxOf(a.rect.left, b.rect.left)
    val yA = maxOf(a.rect.top, b.rect.top)
    val xB = minOf(a.rect.right, b.rect.right)
    val yB = minOf(a.rect.bottom, b.rect.bottom)

    // Compute the area of intersection rectangle
    val interArea = maxOf(0f, xB - xA) * maxOf(0f, yB - yA)

    // Compute the area of both the prediction and ground-truth
    // rectangles
    val box1Area = a.rect.width() * a.rect.height()
    val box2Area = b.rect.width() * b.rect.height()

    // Compute the intersection over union by taking the intersection
    // area and dividing it by the sum of prediction + ground-truth
    // areas - the intersection area
    return if (box1Area + box2Area - interArea == 0f) {
        0f
    } else {
        interArea / (box1Area + box2Area - interArea)
    }
}
