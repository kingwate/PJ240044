package com.example.my_application

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class ModelUtils(private val context: Context) {

    // 类别标签
    private val labels = arrayOf("兽骨", "龟甲")

    // 加载模型
    fun loadModel(): Module {
        try {
            // 将模型从assets复制到本地存储
            val modelFile = File(context.filesDir, "model_mobile.pt")
            if (!modelFile.exists()) {
                context.assets.open("model_mobile.pt").use { inputStream ->
                    FileOutputStream(modelFile).use { outputStream ->
                        val buffer = ByteArray(4 * 1024)
                        var read: Int
                        while (inputStream.read(buffer).also { read = it } != -1) {
                            outputStream.write(buffer, 0, read)
                        }
                    }
                }
            }
            // 加载模型
            return Module.load(modelFile.absolutePath)
        } catch (e: IOException) {
            e.printStackTrace()
            throw RuntimeException("无法加载模型: ${e.message}", e)
        }
    }

    // 从URI加载并处理图像
    fun loadImageFromUri(uri: Uri): Bitmap {
        val inputStream = context.contentResolver.openInputStream(uri)
        val bitmap = BitmapFactory.decodeStream(inputStream)
        inputStream?.close()
        return bitmap
    }

    // 处理图像并进行预测
    fun classify(module: Module, bitmap: Bitmap): String {
        try {
            // 调整图像大小为224x224，与训练时一致
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
            
            // 使用与训练时相同的预处理参数
            // 根据训练代码，使用的均值和标准差都是(0.5, 0.5, 0.5)
            val mean = floatArrayOf(0.5f, 0.5f, 0.5f)
            val std = floatArrayOf(0.5f, 0.5f, 0.5f)
            
            val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                resizedBitmap,
                mean,
                std
            )
            
            // 进行预测
            val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
            val scores = outputTensor.dataAsFloatArray
            
            // 获取预测结果
            val maxScore = scores.withIndex().maxByOrNull { it.value }
            val classIndex = maxScore?.index ?: 0
            
            return labels[classIndex]
        } catch (e: Exception) {
            e.printStackTrace()
            return "识别失败: ${e.message}"
        }
    }
} 