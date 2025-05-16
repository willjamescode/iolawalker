plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace  = "com.example.footfallng"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.footfallng"
        minSdk        = 30
        targetSdk     = 35
        versionCode   = 1
        versionName   = "1.0"

        // build ***one*** 32-bit ABI for your BLU J8L
        ndk { abiFilters.add("armeabi-v7a") }
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        debug {          /* nothing special - pure JVM build */ }
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions { jvmTarget = "11" }

}

dependencies {
    // MetaWear SDK AAR sitting in app/libs/
    implementation(files("libs/metawear-android-api.aar"))
    implementation("com.parse.bolts:bolts-android:1.4.0")
    //implementation("com.mbientlab.bletoolbox:scanner:0.2.4")
    implementation("androidx.fragment:fragment-ktx:1.6.2")
    // Plain AndroidX + Material ⇒ explicit coordinates
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    // Coroutines – optional but harmless
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    // PyTorch dependencies
    implementation("org.pytorch:pytorch_android:1.13.0")
    implementation("org.pytorch:pytorch_android_torchvision:1.13.0")

    // Tests
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
}