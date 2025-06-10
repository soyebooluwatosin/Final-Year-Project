import { Button, Form, Input, message } from "antd"
import { useForm } from "antd/es/form/Form"
import { LoginFields, validation } from "../static"
import { useNavigate } from "react-router-dom"
import { useState } from "react"

const Login = () => {
  const navigate = useNavigate()
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [form] = useForm<LoginFields>()

  const onSubmit = (values: LoginFields) => {
    setIsSubmitting(true)

    // Since there's no actual authentication, just simulate a quick process
    setTimeout(() => {
      console.log("Login values:", values)

      // Show success message
      message.success("Welcome back!")

      // Redirect to Analysis page
      navigate("/analysis")

      setIsSubmitting(false)
    }, 800) // Reduced delay for better UX
  }

  return (
    <div className="bg-primary grid min-h-screen place-items-center">
      <div className="flex min-w-[480px] flex-col items-center gap-5 rounded-xl bg-white/80 p-8">
        <div className="space-y-1 text-center">
          <h2 className="text-3xl font-bold capitalize">Welcome back!</h2>
          <p className="font-semibold text-[#A79A9A]">
            Enter your details to continue
          </p>
        </div>
        <Form
          form={form}
          onFinish={onSubmit}
          requiredMark
          scrollToFirstError
          autoComplete="on"
          layout="vertical"
          name="login"
          className="w-full"
        >
          <Form.Item<LoginFields>
            hasFeedback
            label="Email Address"
            name="email"
            rules={[
              { type: "email", message: validation.email },
              { required: true, message: validation.required },
            ]}
          >
            <Input
              placeholder="email e.g hi@gmail.com"
              className="h-11 !rounded-full"
            />
          </Form.Item>

          <Form.Item<LoginFields>
            hasFeedback
            label="Password"
            name="password"
            rules={[{ required: true, message: validation.password.length }]}
          >
            <Input.Password
              placeholder="Enter password"
              className="h-11 !rounded-full"
            />
          </Form.Item>

          <div className="flex justify-end">
            <Button
              type="link"
              className="text-primary mb-5 !p-0 text-sm hover:cursor-pointer"
              onClick={() =>
                message.info(
                  "Password reset functionality would be implemented here"
                )
              }
            >
              Forgot Password?
            </Button>
          </div>

          <Button
            type="primary"
            htmlType="submit"
            loading={isSubmitting}
            className="!h-11 !w-full !rounded-full !text-base !font-semibold"
          >
            {isSubmitting ? "Logging in..." : "Login"}
          </Button>
        </Form>

        <div className="mt-4 text-center">
          Don't have an account?
          <Button
            type="link"
            onClick={() => navigate("/signup")}
            className="!ml-1 !p-0"
          >
            Sign up
          </Button>
        </div>
      </div>
    </div>
  )
}

export default Login