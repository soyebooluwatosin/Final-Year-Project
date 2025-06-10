import { Button, Spin, Card, Modal, Select, Slider, message, Statistic, Row, Col, Tag, Alert } from "antd"
import { MessageOutlined, LikeOutlined, DislikeOutlined, ReloadOutlined, CheckCircleOutlined, ExclamationCircleOutlined } from "@ant-design/icons"
import CameraComponent from "../components/camera"
import { useState, useEffect } from "react"
import { analyzeEmotion, submitFeedback, EmotionAnalysisResult, FeedbackData } from "../services/api.ts"

const { Option } = Select

const Analysis = () => {
  const [capturedImage, setCapturedImage] = useState<string | null>(null)
  const [analyzing, setAnalyzing] = useState<boolean>(false)
  const [analysisResult, setAnalysisResult] = useState<EmotionAnalysisResult | null>(null)
  const [analysisError, setAnalysisError] = useState<string | null>(null)
  const [apiStatus, setApiStatus] = useState<'checking' | 'connected' | 'error'>('checking')
  
  // Feedback states
  const [feedbackModalVisible, setFeedbackModalVisible] = useState<boolean>(false)
  const [feedbackEmotion, setFeedbackEmotion] = useState<string>("")
  const [feedbackStress, setFeedbackStress] = useState<number>(5.0)
  const [submittingFeedback, setSubmittingFeedback] = useState<boolean>(false)

  const emotionOptions = [
    "angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"
  ]

  // Check API status on component mount
  useEffect(() => {
    const checkAPI = async () => {
      console.log("🔍 Checking API status...")
      try {
        const response = await fetch('http://127.0.0.1:8000/', {
          method: 'GET',
        })
        if (response.ok) {
          const data = await response.json()
          console.log("✅ API is running:", data)
          setApiStatus('connected')
        } else {
          console.log("❌ API responded with error:", response.status)
          setApiStatus('error')
        }
      } catch (error) {
        console.log("❌ API connection failed:", error)
        setApiStatus('error')
      }
    }
    
    checkAPI()
  }, [])

  const handleImageCapture = async (imageData: string) => {
    console.log("🖼️ ===== IMAGE CAPTURE DEBUG =====")
    console.log("🖼️ Image captured, length:", imageData.length)
    console.log("🖼️ Image format:", imageData.substring(0, 50))
    console.log("🖼️ Setting captured image state...")
    
    setCapturedImage(imageData)
    setAnalyzing(true)
    setAnalysisError(null)
    setAnalysisResult(null)

    console.log("🔄 ===== API CALL DEBUG =====")
    console.log("🔄 Starting analysis...")
    console.log("🔄 API Status:", apiStatus)

    try {
      console.log("🔄 Calling analyzeEmotion API...")
      console.log("🔄 Image data length being sent:", imageData.length)
      
      const result = await analyzeEmotion(imageData)
      
      console.log("✅ ===== API RESPONSE DEBUG =====")
      console.log("✅ Analysis result received:", result)
      console.log("✅ Result success:", result.success)
      console.log("✅ Result emotion:", result.emotion)
      console.log("✅ Result confidence:", result.confidence)
      console.log("✅ Result stress level:", result.stressLevel)
      
      if (result.success) {
        setAnalysisResult(result)
        message.success("Analysis completed successfully!")
        console.log("✅ Analysis result set in state")
      } else {
        const errorMsg = result.message || "Analysis failed"
        console.log("❌ Analysis marked as failed:", errorMsg)
        throw new Error(errorMsg)
      }
    } catch (error) {
      console.log("❌ ===== ERROR DEBUG =====")
      console.error("❌ Analysis failed with error:", error)
      console.error("❌ Error type:", typeof error)
      console.error("❌ Error message:", error instanceof Error ? error.message : String(error))
      
      const errorMessage = error instanceof Error ? error.message : "Analysis failed. Please try again."
      setAnalysisError(errorMessage)
      message.error(errorMessage)
    } finally {
      console.log("🏁 Analysis process completed, setting analyzing to false")
      setAnalyzing(false)
    }
  }

  const handlePositiveFeedback = () => {
    if (analysisResult && capturedImage) {
      const feedbackData: FeedbackData = {
        image: capturedImage,
        predicted_emotion: analysisResult.emotion,
        predicted_stress: analysisResult.stressLevel,
        predicted_confidence: analysisResult.confidence / 100,
        correct_emotion: analysisResult.emotion,
        correct_stress: analysisResult.stressLevel,
        user_id: `user_${Date.now()}`
      }
      
      submitFeedbackData(feedbackData)
    }
  }

  const handleNegativeFeedback = () => {
    if (analysisResult) {
      setFeedbackEmotion(analysisResult.emotion)
      setFeedbackStress(analysisResult.stressLevel)
      setFeedbackModalVisible(true)
    }
  }

  const submitFeedbackData = async (feedbackData: FeedbackData) => {
    setSubmittingFeedback(true)
    try {
      const result = await submitFeedback(feedbackData)
      if (result.success) {
        message.success(
          `Thank you for your feedback! Reward: ${result.reward.toFixed(2)}. Total feedback: ${result.total_feedback}`
        )
        
        // Update the analysis result with new feedback count
        if (analysisResult) {
          setAnalysisResult({
            ...analysisResult,
            feedback_count: result.total_feedback
          })
        }
      } else {
        message.error("Failed to submit feedback")
      }
    } catch (error) {
      console.error("Feedback submission failed:", error)
      message.error("Failed to submit feedback")
    } finally {
      setSubmittingFeedback(false)
    }
  }

  const handleCorrectiveFeedback = () => {
    if (analysisResult && capturedImage) {
      const feedbackData: FeedbackData = {
        image: capturedImage,
        predicted_emotion: analysisResult.emotion,
        predicted_stress: analysisResult.stressLevel,
        predicted_confidence: analysisResult.confidence / 100,
        correct_emotion: feedbackEmotion,
        correct_stress: feedbackStress,
        user_id: `user_${Date.now()}`
      }
      
      submitFeedbackData(feedbackData)
      setFeedbackModalVisible(false)
    }
  }

  const handleReset = () => {
    console.log("🔄 Resetting component state...")
    setCapturedImage(null)
    setAnalysisResult(null)
    setAnalysisError(null)
    setFeedbackModalVisible(false)
  }

  const getStressColor = (level: number) => {
    if (level < 3) return "#52c41a"
    if (level < 6) return "#faad14"
    if (level < 8) return "#fa8c16"
    return "#f5222d"
  }

  const getEmotionColor = (emotion: string) => {
    const colors: Record<string, string> = {
      happy: "#52c41a",
      neutral: "#1890ff", 
      sad: "#722ed1",
      angry: "#f5222d",
      fear: "#fa8c16",
      surprise: "#13c2c2",
      disgust: "#eb2f96",
      contempt: "#faad14"
    }
    return colors[emotion] || "#666666"
  }

  return (
    <div className="min-h-screen space-y-18">
      <div className="bg-primary font-kaushan h-[109px]">
        <div className="max-width-responsive flex h-full items-center justify-between">
          <p className="text-7xl text-white capitalize">Stress analysis</p>
          <p className="text-5xl uppercase">nisot</p>
        </div>
      </div>



      <div className="max-width-responsive grid grid-cols-2 gap-8">
        {/* Camera Section */}
        <div className="border-r-primary border-r-2 pr-10">
          <CameraComponent setCapturedImage={handleImageCapture} />

          {capturedImage && (
            <Card className="mt-4" size="small" title="Captured Image">
              <img
                src={capturedImage}
                alt="Captured"
                className="max-w-full rounded shadow-md"
              />
              <div className="mt-2 text-xs text-gray-500">
                Image size: {capturedImage.length} bytes
              </div>
            </Card>
          )}
        </div>

        {/* Analysis Results Section */}
        <div className="space-y-5 pl-10">
          <h2 className="text-center text-3xl font-semibold">Analysis Results</h2>



          {analyzing ? (
            <div className="flex items-center justify-center py-10">
              <Spin size="large" tip="Analyzing facial expression..." />
            </div>
          ) : (
            <>
              {analysisError && (
                <Alert
                  type="error"
                  message="Analysis Failed"
                  description={analysisError}
                  showIcon
                  className="mb-4"
                />
              )}

              {analysisResult ? (
                <div className="space-y-6">
                  {/* Main Results */}
                  <Card>
                    <Row gutter={16}>
                      <Col span={12}>
                        <Statistic 
                          title="Detected Emotion" 
                          value={analysisResult.emotion}
                          valueStyle={{ 
                            color: getEmotionColor(analysisResult.emotion),
                            textTransform: 'capitalize'
                          }}
                        />
                        <div className="mt-2">
                          <Tag color={getEmotionColor(analysisResult.emotion)}>
                            {analysisResult.confidence.toFixed(1)}% confidence
                          </Tag>
                          {analysisResult.correction_applied && (
                            <Tag color="blue">AI Corrected</Tag>
                          )}
                        </div>
                      </Col>
                      <Col span={12}>
                        <Statistic 
                          title="Stress Level" 
                          value={`${analysisResult.stressLevel.toFixed(1)}/10`}
                          valueStyle={{ color: getStressColor(analysisResult.stressLevel) }}
                        />
                        <div className="mt-2">
                          <Tag color={getStressColor(analysisResult.stressLevel)}>
                            {analysisResult.stressCategory}
                          </Tag>
                        </div>
                      </Col>
                    </Row>
                  </Card>

                  {/* Top Emotions */}
                  {analysisResult.allEmotions && (
                    <Card title="Emotion Breakdown" size="small">
                      <div className="space-y-2">
                        {Object.entries(analysisResult.allEmotions)
                          .sort(([,a], [,b]) => b - a)
                          .slice(0, 3)
                          .map(([emotion, value]) => (
                            <div key={emotion} className="flex justify-between items-center">
                              <span className="capitalize">{emotion}:</span>
                              <Tag color={getEmotionColor(emotion)}>
                                {(value * 100).toFixed(1)}%
                              </Tag>
                            </div>
                          ))}
                      </div>
                    </Card>
                  )}

                  {/* Suggestions */}
                  <Card title="Suggestions" size="small">
                    {analysisResult.suggestions && analysisResult.suggestions.length > 0 ? (
                      <ul className="list-disc pl-5 space-y-1">
                        {analysisResult.suggestions.map((suggestion, index) => (
                          <li key={index} className="text-sm">{suggestion}</li>
                        ))}
                      </ul>
                    ) : (
                      <p className="text-gray-500">No suggestions available</p>
                    )}
                  </Card>

                  {/* Feedback Section */}
                  <Card title={
                    <div className="flex items-center gap-2">
                      <MessageOutlined />
                      <span>Help improve the model</span>
                    </div>
                  }>
                    <p className="mb-4 text-sm text-gray-600">
                      Was this analysis accurate? Your feedback helps improve our AI model.
                    </p>
                    <div className="flex gap-3">
                      <Button
                        type="primary"
                        icon={<LikeOutlined />}
                        onClick={handlePositiveFeedback}
                        loading={submittingFeedback}
                        className="flex-1"
                      >
                        Correct
                      </Button>
                      <Button
                        danger
                        icon={<DislikeOutlined />}
                        onClick={handleNegativeFeedback}
                        loading={submittingFeedback}
                        className="flex-1"
                      >
                        Incorrect
                      </Button>
                    </div>
                    <div className="mt-3 flex justify-between items-center text-xs text-gray-500">
                      <span>Total feedback received: {analysisResult.feedback_count}</span>
                      <span>Help us improve with your input!</span>
                    </div>
                  </Card>
                </div>
              ) : !capturedImage ? (
                <Card>
                  <div className="text-center text-gray-500 py-8">
                    <p className="text-lg">Take a picture to start analysis</p>
                    <p className="text-sm mt-2">Position your face in the camera and click "Take Picture"</p>
                  </div>
                </Card>
              ) : (
                <Card>
                  <div className="text-center text-gray-500 py-8">
                    <p className="text-lg">Image captured</p>
                    <p className="text-sm mt-2">Click "Take Picture" again to analyze...</p>
                  </div>
                </Card>
              )}
            </>
          )}

          {/* Reset Button */}
          <div className="flex justify-center">
            <Button
              type="primary"
              icon={<ReloadOutlined />}
              className="!rounded-lg !px-14 !py-5 !text-xl"
              onClick={handleReset}
              disabled={analyzing}
            >
              Reset
            </Button>
          </div>
        </div>
      </div>

      {/* Feedback Correction Modal */}
      <Modal
        title="Correct the Analysis"
        open={feedbackModalVisible}
        onOk={handleCorrectiveFeedback}
        onCancel={() => setFeedbackModalVisible(false)}
        confirmLoading={submittingFeedback}
        okText="Submit Correction"
      >
        <div className="space-y-4">
          <p className="text-sm text-gray-600">
            Please provide the correct emotion and stress level to help us improve our AI.
          </p>
          
          <div>
            <label className="block text-sm font-medium mb-2">Correct Emotion:</label>
            <Select
              value={feedbackEmotion}
              onChange={setFeedbackEmotion}
              className="w-full"
              placeholder="Select the correct emotion"
            >
              {emotionOptions.map(emotion => (
                <Option key={emotion} value={emotion}>
                  <span className="capitalize">{emotion}</span>
                </Option>
              ))}
            </Select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">
              Correct Stress Level: {feedbackStress.toFixed(1)}/10
            </label>
            <Slider
              min={0}
              max={10}
              step={0.1}
              value={feedbackStress}
              onChange={setFeedbackStress}
              marks={{
                0: 'No Stress',
                2.5: 'Low',
                5: 'Moderate', 
                7.5: 'High',
                10: 'Severe'
              }}
            />
          </div>

          <div className="text-xs text-gray-500">
            Your feedback is anonymous and helps improve the accuracy of our emotion recognition system.
          </div>
        </div>
      </Modal>
    </div>
  )
}

export default Analysis