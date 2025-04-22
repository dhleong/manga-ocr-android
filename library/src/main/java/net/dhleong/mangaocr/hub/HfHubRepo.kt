package net.dhleong.mangaocr.hub

import android.content.Context
import android.net.Uri
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.net.URL
import java.security.MessageDigest

class HfHubRepo(
    val name: String,
    private val endpoint: String = "https://huggingface.co",
) {
    suspend fun resolveLocalPath(
        context: Context,
        repoPath: String,
        revision: String = "main",
        sha256: String? = null,
    ): File {
        val uri =
            Uri
                .parse(endpoint)
                .buildUpon()
                .appendPath(name)
                .appendPath("resolve")
                .appendPath(revision)
                .appendPath(repoPath)
                .build()

        val parent = File(File(context.noBackupFilesDir, "hf"), name)
        val localPath = File(parent, repoPath)
        if (localPath.exists()) {
            return localPath
        }

        Log.v("HUB", "Downloading $repoPath")

        // NOTE: localPath.parentFile *may* not be the same as parent
        requireNotNull(localPath.parentFile).mkdirs()

        val tmpPath = File(parent, "$repoPath.tmp")
        localPath.delete()
        tmpPath.delete()

        withContext(Dispatchers.IO) {
            // TODO: OkHttp?
            URL(uri.toString()).openStream().buffered().use { input ->
                tmpPath.outputStream().buffered().use { out ->
                    input.copyTo(out)
                }
            }

            Log.v("HUB", "Downloaded ${tmpPath.length()} bytes to $tmpPath")

            val actual = getSha256Hash(tmpPath)
            Log.v("HUB", "Actual hash = $actual")

            if (sha256 != null && actual != sha256) {
                // verify sha
                throw RuntimeException("Sha mismatch for $repoPath: got $actual expected $sha256")
            }
            tmpPath.renameTo(localPath)
        }

        return localPath
    }

    @OptIn(ExperimentalStdlibApi::class)
    private suspend fun getSha256Hash(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")

        withContext(Dispatchers.IO) {
            file.inputStream().buffered().use { input ->
                val bytes = ByteArray(8192)
                while (true) {
                    val read = input.read(bytes)
                    if (read > 0) {
                        md.update(bytes, 0, read)
                    } else {
                        break
                    }
                }
            }
        }

        val digest = md.digest()
        return digest.toHexString()
    }
}
