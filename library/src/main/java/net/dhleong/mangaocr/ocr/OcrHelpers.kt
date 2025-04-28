package net.dhleong.mangaocr.ocr

import java.nio.FloatBuffer

fun FloatBuffer.findMaxArg(
    start: Int,
    end: Int,
): Int {
    // find argmax
    var maxTokenId = -1
    var maxArg = 0f
    for (i in start until end) {
        val value = get(i)
        if (maxTokenId < 0 || value > maxArg) {
//                    Log.v("ORT", "max @$i token $i = $value")
            maxTokenId = (i - start)
//                    maxTokenId = candidateTokenId
            maxArg = value
        }
//                candidateTokenId += 1
    }

    if (maxTokenId < 0) {
        throw IllegalStateException("No max token found")
    }

    return maxTokenId
}
