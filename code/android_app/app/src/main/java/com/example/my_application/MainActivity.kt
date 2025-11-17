package com.example.my_application

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import coil.compose.AsyncImage
import com.example.my_application.ui.theme.My_ApplicationTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.pytorch.Module
import java.io.File
import java.util.*
import androidx.compose.foundation.clickable

class MainActivity : ComponentActivity() {
    companion object {
        private const val TAG = "MainActivity"
    }
    
    private var imageUri: Uri? = null
    private var imageBitmap: Bitmap? = null
    private var displayText by mutableStateOf("请点击下方按钮选择图片")
    private var isProcessing by mutableStateOf(false)
    
    // 模型工具类
    private lateinit var modelUtils: ModelUtils
    // PyTorch模型
    private var module: Module? = null

    private val requestCameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            takePicture()
        } else {
            displayText = "需要相机权限才能拍照"
        }
    }

    private val requestStoragePermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            pickImage()
        } else {
            displayText = "需要存储权限才能选择图片"
        }
    }

    private val takePictureLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success ->
        if (success) {
            imageUri?.let { uri ->
                processImageWithModel(uri)
            }
        }
    }

    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        if (uri != null) {
            imageUri = uri
            processImageWithModel(uri)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // 初始化模型工具类
        modelUtils = ModelUtils(this)
        
        // 设置初始显示文本
        displayText = "请点击下方按钮选择图片"
        
        // 在后台线程加载模型
        Thread {
            try {
                Log.d(TAG, "开始加载模型...")
                module = modelUtils.loadModel()
                Log.d(TAG, "模型加载成功")
            } catch (e: Exception) {
                Log.e(TAG, "模型加载失败", e)
                runOnUiThread {
                    displayText = "模型加载失败: ${e.message}"
                }
            }
        }.start()
        
        enableEdgeToEdge()
        setContent {
            My_ApplicationTheme {
                MainScreen(
                    onTakePicture = { /* 不再使用 */ },
                    onPickImage = { checkStoragePermission() },
                    imageUri = imageUri,
                    displayText = displayText,
                    isProcessing = isProcessing
                )
            }
        }
    }

    private fun processImageWithModel(uri: Uri) {
        isProcessing = true
        displayText = "识别中..."
        
        Thread {
            try {
                // 确保模型已加载
                if (module == null) {
                    Log.d(TAG, "模型未加载，尝试加载模型...")
                    module = modelUtils.loadModel()
                }
                
                // 加载图像
                Log.d(TAG, "加载图像...")
                val bitmap = modelUtils.loadImageFromUri(uri)
                imageBitmap = bitmap
                
                // 模拟处理延迟
                Thread.sleep(1000)
                
                // 使用模型进行分类
                Log.d(TAG, "开始进行图像分类...")
                val result = modelUtils.classify(module!!, bitmap)
                Log.d(TAG, "分类结果: $result")
                
                // 更新UI
                runOnUiThread {
                    displayText = result
                    isProcessing = false
                }
            } catch (e: Exception) {
                Log.e(TAG, "处理图像时出错", e)
                runOnUiThread {
                    displayText = "识别失败: ${e.message}"
                    isProcessing = false
                }
            }
        }.start()
    }

    private fun checkCameraPermission() {
        when {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> {
                takePicture()
            }
            else -> {
                requestCameraPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }

    private fun checkStoragePermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            when {
                ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.READ_MEDIA_IMAGES
                ) == PackageManager.PERMISSION_GRANTED -> {
                    pickImage()
                }
                else -> {
                    requestStoragePermissionLauncher.launch(Manifest.permission.READ_MEDIA_IMAGES)
                }
            }
        } else {
            when {
                ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.READ_EXTERNAL_STORAGE
                ) == PackageManager.PERMISSION_GRANTED -> {
                    pickImage()
                }
                else -> {
                    requestStoragePermissionLauncher.launch(Manifest.permission.READ_EXTERNAL_STORAGE)
                }
            }
        }
    }

    private fun takePicture() {
        try {
            val imageFile = File.createTempFile(
                "JPEG_${System.currentTimeMillis()}_",
                ".jpg",
                externalCacheDir
            )
            val uri = FileProvider.getUriForFile(
                this,
                "${packageName}.fileprovider",
                imageFile
            )
            imageUri = uri
            takePictureLauncher.launch(uri)
        } catch (e: Exception) {
            Log.e(TAG, "拍照过程中出错", e)
            displayText = "无法启动相机: ${e.message}"
        }
    }

    private fun pickImage() {
        try {
            pickImageLauncher.launch("image/*")
        } catch (e: Exception) {
            Log.e(TAG, "选择图片过程中出错", e)
            displayText = "无法选择图片: ${e.message}"
        }
    }
}

@Composable
fun MainScreen(
    onTakePicture: () -> Unit,
    onPickImage: () -> Unit,
    imageUri: Uri?,
    displayText: String,
    isProcessing: Boolean
) {
    val blueGradient = Brush.verticalGradient(
        colors = listOf(
            Color(0xFF64B5F6),
            Color(0xFF42A5F5),
            Color(0xFF1E88E5)
        )
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(blueGradient)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(24.dp)
        ) {
            // 应用标题
            Text(
                text = "甲骨材质识别",
                fontSize = 28.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White,
                modifier = Modifier.padding(top = 32.dp, bottom = 16.dp)
            )
            
            // 图片显示卡片
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(320.dp)
                    .shadow(8.dp, RoundedCornerShape(16.dp)),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = Color.White)
            ) {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    if (imageUri != null) {
                        Box(
                            modifier = Modifier.fillMaxSize(),
                            contentAlignment = Alignment.Center
                        ) {
                            // 添加背景
                            Box(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .background(Color.White)
                            )
                            // 显示图片
                            AsyncImage(
                                model = imageUri,
                                contentDescription = "选择的图片",
                                modifier = Modifier
                                    .fillMaxSize()
                                    .padding(8.dp),
                                contentScale = ContentScale.Fit
                            )
                        }
                    } else {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.Center
                        ) {
                            Icon(
                                painter = painterResource(id = android.R.drawable.ic_menu_gallery),
                                contentDescription = "图片图标",
                                tint = Color(0xFF1E88E5),
                                modifier = Modifier.size(64.dp)
                            )
                            Spacer(modifier = Modifier.height(16.dp))
                            Text(
                                text = "请选择图片",
                                color = Color(0xFF757575),
                                fontSize = 18.sp
                            )
                        }
                    }
                }
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            
            // 从相册选择按钮
            Button(
                onClick = onPickImage,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp)
                    .padding(horizontal = 32.dp),
                enabled = !isProcessing,
                shape = RoundedCornerShape(28.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFFF5F5DC),
                    contentColor = Color(0xFF333333),
                    disabledContainerColor = Color(0xFFF5F5DC).copy(alpha = 0.7f),
                    disabledContentColor = Color(0xFF333333).copy(alpha = 0.5f)
                )
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.Center
                ) {
                    Icon(
                        painter = painterResource(id = android.R.drawable.ic_menu_gallery),
                        contentDescription = "图片选择",
                        modifier = Modifier.size(24.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "从相册选择",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(24.dp))
            
            // 结果显示区域
            AnimatedVisibility(
                visible = true,
                enter = fadeIn(animationSpec = tween(500)) + 
                        slideInVertically(animationSpec = tween(500)) { it / 2 },
                exit = fadeOut(animationSpec = tween(500))
            ) {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = if (isProcessing) Color(0xFFE8F5E9) else Color(0xFFE3F2FD)
                    )
                ) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(24.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        if (isProcessing) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.Center
                            ) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(24.dp),
                                    color = Color(0xFF66BB6A)
                                )
                                Spacer(modifier = Modifier.width(16.dp))
                                Text(
                                    text = displayText,
                                    fontSize = 18.sp,
                                    color = Color(0xFF66BB6A)
                                )
                            }
                        } else {
                            Text(
                                text = displayText,
                                fontSize = 22.sp,
                                fontWeight = FontWeight.Bold,
                                color = if (displayText == "龟甲" || displayText == "兽骨") 
                                          Color(0xFF1565C0) else Color(0xFF757575)
                            )
                        }
                    }
                }
            }
            
            // 底部信息
            Spacer(modifier = Modifier.weight(1f))
            Text(
                text = "© 2025 甲骨材质识别系统",
                fontSize = 12.sp,
                color = Color.White.copy(alpha = 0.7f),
                modifier = Modifier.padding(bottom = 16.dp)
            )
        }
    }
}

@Preview(showBackground = true)
@Composable
fun MainScreenPreview() {
    My_ApplicationTheme {
        MainScreen(
            onTakePicture = {},
            onPickImage = {},
            imageUri = null,
            displayText = "请选择图片",
            isProcessing = false
        )
    }
}
