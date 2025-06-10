import { ConfigProviderProps, ConfigProvider } from "antd";

export const AntdConfigProvider = (props: ConfigProviderProps) => {
  return (
    <ConfigProvider
      theme={{
        token: {
          fontFamily: "inherit",
          blue5: "#DB8469",
          colorPrimary: "#DB8469",
          colorError: "#D64751",
          colorSuccess: "#DB8469",
          colorWarning: "#6c6727",
        },
        components: {
          Button: {
            colorPrimary: "#DB8469",
            colorPrimaryHover: "#DB8469",
            blue5: "#DB8469",
            controlOutline: "#DB8469",
            defaultActiveColor: "#DB8469",
            screenSM: 640,
            borderRadius: 6,
            defaultActiveBg: "#DB8469",
          },
          Input: {
            controlOutline: "#DB8469",
            colorErrorOutline: "#D64751",
            colorError: "#D64751",
            controlOutlineWidth: 1,
            colorBorder: "#909090",
            hoverBorderColor: "#DB8469",
            colorTextPlaceholder: "#9F9F9F",
            padding: 14,
            paddingInline: 20,
            borderRadius: 8,
            colorBgContainer: "transparent",
            colorText: "#101828",
            fontSize: 16,
            addonBg: "#F3F4F6",
          },
          Form: {
            itemMarginBottom: 24,
            labelColor: "#292929",
            labelFontSize: 16,
          },
          Message: {
            colorBgBase: "#ffffff",
            colorText: "#292929",
          },
        },
      }}
      {...props}
    >
      {props.children}
    </ConfigProvider>
  );
};
