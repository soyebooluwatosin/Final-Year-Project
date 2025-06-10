import { useState } from "react"
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from "recharts"

// Define the interface for the component props
interface EmotionChartProps {
  emotionData: Record<string, number>
}

// Define the interface for the data structure
interface EmotionDataPoint {
  name: string
  value: number
}

const EmotionChart: React.FC<EmotionChartProps> = ({ emotionData }) => {
  const [activeIndex, setActiveIndex] = useState<number | null>(null)

  // Filter emotions with values > 1%
  const filteredData: EmotionDataPoint[] = Object.entries(emotionData || {})
    .filter(([_, value]) => value > 0.01)
    .map(([name, value]) => ({
      name,
      value: Number(value) * 100,
    }))
    .sort((a, b) => b.value - a.value)

  const COLORS = [
    "#DB8469", // Primary color for top emotion
    "#FFB347", // Orange
    "#A5D7E8", // Light blue
    "#D7BDE2", // Lavender
    "#ABEBC6", // Light green
    "#F5CBA7", // Light orange
    "#AED6F1", // Sky blue
    "#F9E79F", // Light yellow
  ]

  const onPieEnter = (_: any, index: number) => {
    setActiveIndex(index)
  }

  const onPieLeave = () => {
    setActiveIndex(null)
  }

  const renderLabel = (entry: EmotionDataPoint) => {
    return `${entry.name}: ${entry.value.toFixed(1)}%`
  }

  if (!emotionData || Object.keys(emotionData).length === 0) {
    return <div>No emotion data available</div>
  }

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={filteredData}
            cx="50%"
            cy="50%"
            innerRadius={60}
            outerRadius={80}
            paddingAngle={2}
            dataKey="value"
            label={renderLabel}
            onMouseEnter={onPieEnter}
            onMouseLeave={onPieLeave}
          >
            {filteredData.map((_, index) => (
              <Cell
                key={`cell-${index}`}
                fill={COLORS[index % COLORS.length]}
                stroke="#fff"
                strokeWidth={index === activeIndex ? 2 : 1}
                style={{
                  filter:
                    index === activeIndex
                      ? "drop-shadow(0px 0px 4px rgba(0, 0, 0, 0.3))"
                      : "none",
                  cursor: "pointer",
                  opacity:
                    activeIndex === null || index === activeIndex ? 1 : 0.7,
                }}
              />
            ))}
          </Pie>
          <Tooltip formatter={(value: number) => `${value.toFixed(1)}%`} />
          <Legend verticalAlign="bottom" height={36} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  )
}

export default EmotionChart
