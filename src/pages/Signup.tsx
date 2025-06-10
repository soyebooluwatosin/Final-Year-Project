import { Button, Form, Input, message } from "antd"
import { SignupFields, validation } from "../static"
import { useForm } from "antd/es/form/Form"
import { useNavigate } from "react-router-dom"
import { useState } from "react"

const Signup = () => {
  const navigate = useNavigate()
  const [isSubmitting, setIsSubmitting] = useState(false)

  const validatePassword = (_: any, value: string) => {
    if (!value) {
      return Promise.reject("Password is required")
    }

    const hasUppercase = /[A-Z]/.test(value)
    const hasLowercase = /[a-z]/.test(value)
    const hasNumber = /\d/.test(value)
    const hasSpecialChar = /[\W_]/.test(value) // Non-word character or underscore

    if (!hasUppercase)
      return Promise.reject("Password must include an uppercase letter")
    if (!hasLowercase)
      return Promise.reject("Password must include a lowercase letter")
    if (!hasNumber) return Promise.reject("Password must include a number")
    if (!hasSpecialChar)
      return Promise.reject("Password must include a special character")

    return Promise.resolve()
  }

  const [form] = useForm<SignupFields>()

  const onSubmit = (values: SignupFields) => {
    setIsSubmitting(true)

    // Since there's no actual registration, just simulate a quick process
    setTimeout(() => {
      console.log("Signup values:", values)

      // Show success message
      message.success("Account created successfully! Welcome to nIsoT!")

      // Redirect to Analysis page
      navigate("/analysis")

      setIsSubmitting(false)
    }, 800) // Reduced delay for better UX
  }

  return (
    <div className="bg-primary grid min-h-screen place-items-center">
      <div className="flex min-w-[500px] flex-col items-center gap-5 rounded-xl bg-white/80 p-8">
        <div className="space-y-1 text-center">
          <h2 className="text-3xl font-bold capitalize">Welcome!</h2>
          <p className="font-semibold text-[#A79A9A]">
            Create your account to get started
          </p>
        </div>
        <Form
          form={form}
          onFinish={onSubmit}
          requiredMark
          scrollToFirstError
          autoComplete="on"
          layout="vertical"
          name="signup"
          className="w-full"
        >
          <Form.Item<SignupFields>
            hasFeedback
            label="Full Name"
            name="full_name"
            rules={[{ required: true, message: validation.required }]}
          >
            <Input
              placeholder="Enter your full name"
              className="h-11 !rounded-full"
            />
          </Form.Item>

          <Form.Item<SignupFields>
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

          <Form.Item<SignupFields>
            hasFeedback
            label="Password"
            name="password"
            validateTrigger="onChange"
            rules={[{ validator: validatePassword }]}
          >
            <Input.Password
              placeholder="Enter password"
              className="h-11 !rounded-full"
            />
          </Form.Item>

          <Form.Item<SignupFields>
            hasFeedback
            label="Confirm Password"
            name="confirm_password"
            dependencies={["password"]}
            rules={[
              { required: true, message: validation.required },
              ({ getFieldValue }) => ({
                validator(_, value) {
                  if (!value || getFieldValue("password") === value) {
                    return Promise.resolve()
                  }
                  return Promise.reject(new Error(validation.password.noMatch))
                },
              }),
            ]}
          >
            <Input.Password
              placeholder="Re-enter password"
              className="h-11 !rounded-full"
            />
          </Form.Item>

          <Button
            htmlType="submit"
            type="primary"
            loading={isSubmitting}
            className="!h-11 !w-full !rounded-full !text-base !font-semibold"
          >
            {isSubmitting ? "Creating Account..." : "Sign up"}
          </Button>
        </Form>

        <div className="mt-4 text-center">
          Already have an account?
          <Button
            type="link"
            onClick={() => navigate("/login")}
            className="!ml-1 !p-0"
          >
            Log in
          </Button>
        </div>
      </div>
    </div>
  )
}

export default Signup
