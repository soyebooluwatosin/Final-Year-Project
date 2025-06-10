export const APP_NAME = "NISOT"

export const validation = {
  required: "*This field is required",
  email: "*Please enter a valid email address",
  number: "*Please enter a valid number",
  password: {
    length: "*Password must be at least 8 characters",
    number: "*Password must have a number",
    specialChar: "*Password must have a special character",
    noMatch: "*Passwords do not match",
  },
}

export type LoginFields = {
    email: string
    password: string
}

export type SignupFields = {
    full_name: string
    email: string
    password: string
    confirm_password: string
}