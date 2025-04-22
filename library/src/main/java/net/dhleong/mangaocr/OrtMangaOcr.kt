package net.dhleong.mangaocr

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import net.dhleong.mangaocr.hub.HfHubRepo

class OrtMangaOcr private constructor(
    session: OrtSession,
) : MangaOcr {
    override suspend fun process() {
        TODO("Not yet implemented")
    }

    companion object {
        suspend fun initialize(context: Context): OrtMangaOcr {
            val ctx = context.applicationContext
            val env = OrtEnvironment.getEnvironment()
            val repo = HfHubRepo("mayocream/koharu")
            val (model, vocab) =
                coroutineScope {
                    awaitAll(
                        async {
                            repo.resolveLocalPath(
                                ctx,
                                "manga-ocr.onnx",
                                sha256 = "ac11c392af15df90f07d9a6473c737b7c18fdd31af6756768b6a7886f1fb3be1",
                            )
                        },
                        async {
                            repo.resolveLocalPath(
                                ctx,
                                "vocab.txt",
                                sha256 = "344fbb6b8bf18c57839e924e2c9365434697e0227fac00b88bb4899b78aa594d",
                            )
                        },
                    )
                }
            Log.v("ORT", "Downloaded $model and $vocab")

            val session =
                env.createSession(
                    model.absolutePath,
                    OrtSession.SessionOptions().apply {
                        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.EXTENDED_OPT)
                        setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE)
                        setIntraOpNumThreads(4)
                    },
                )
            Log.v("ORT", "input= ${session.inputNames}")
            return OrtMangaOcr(session)
        }
    }
}
