// Top-level build file where you can add configuration options common to all sub-projects/modules.
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.android.library) apply false
    alias(libs.plugins.kotlin.android) apply false
    alias(libs.plugins.kotlin.compose) apply false
}

val poetry = "${System.getenv("HOME")}/.local/bin/poetry"

val installPoetry =
    tasks.register<Exec>("bootstrapModelInstallPoetry") {
        group = "custom"
        description = "Install python poetry"
        commandLine = listOf("bash", "-c", "curl -sSL 'https://install.python-poetry.org' | python3 -")
    }

val modelDependencies =
    tasks.register<Exec>("bootstrapModelDependencies") {
        group = "custom"
        description = "Run poetry install"
        commandLine = listOf(poetry, "install")
        dependsOn(installPoetry)
    }

tasks.register("bootstrapModel") {
    group = "custom"
    description = "Bootstrap environment for developing the model"
    dependsOn(modelDependencies)
}
