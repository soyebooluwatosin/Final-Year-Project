import { Button } from "antd"
import LandingImage from "../assets/landing.svg?react"
import { Link, useNavigate } from "react-router-dom"

const Landing = () => {
  const navigate = useNavigate()

  return (
    <div className="bg-primary/10 grid max-h-screen min-h-screen grid-cols-1 gap-10 px-10 text-black md:grid-cols-2">
      <div className="order-2 flex h-screen max-h-screen items-center justify-center overflow-hidden md:order-1">
        <LandingImage className="h-full w-auto object-contain" />
      </div>
      <div className="order-1 flex flex-col justify-center gap-5 md:order-2">
        <h1 className="font-kaushan text-4xl font-semibold uppercase md:text-6xl">
          nIsoT
        </h1>
        <h3 className="font-kaushan text-xl font-extralight first-letter:capitalize">
          facial emotion recognition system
        </h3>
        <p className="text-lg font-light md:text-xl">
          Discover your emotional state through advanced facial recognition
          technology. Our system analyzes your facial expressions to detect
          emotions and provides personalized stress management suggestions to
          help you maintain emotional balance.
        </p>
        <Button
          type="primary"
          className="!h-11 !w-fit !px-10 !text-lg"
          onClick={() => navigate("/signup")}
        >
          Get Started
        </Button>
        <div className="italic">
          Already have an account?{" "}
          <Link to={"/login"} className="text-primary font-semibold underline">
            Login
          </Link>
        </div>
      </div>
    </div>
  )
}

export default Landing
