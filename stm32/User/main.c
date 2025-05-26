#include "stm32f10x.h"                  // Device header
#include "Delay.h"
#include "OLED.h"
#include "Serial.h"
#include "LED.h"
uint8_t RxData;			//定义用于接收串口数据的变量

int main(void)
{
	/*模块初始化*/
	OLED_Init();		//OLED初始化
	
	/*显示静态字符串*/
	OLED_ShowString(1, 1, "RxData1:");
	OLED_ShowString(2, 1, "RxData2:");
	uint8_t p1 = GPIO_ReadInputDataBit(GPIOA, GPIO_Pin_1); // 读取PA0电平[3,6](@ref)'
	OLED_ShowNum(1, 9, p1,1);
	uint8_t p2 = GPIO_ReadInputDataBit(GPIOA, GPIO_Pin_2); // 读取PA0电平[3,6](@ref)'
	OLED_ShowNum(2, 9, p2,1);
	/*串口初始化*/
	Serial_Init();		//串口初始化
	LED_Init();
	while (1)
	{
		if (Serial_GetRxFlag() == 1)			//检查串口接收数据的标志位
		{
			RxData = Serial_GetRxData();		//获取串口接收的数据
			Serial_SendByte(RxData);			//串口将收到的数据回传回去，用于测试
			if(RxData==0){
				LED1_ON();
				Delay_ms(20);
				LED1_OFF();
			}
			if(RxData==1){
				LED1_OFF();
			}
			if(RxData==0 || RxData==1){
			
			}
			if(RxData==2){
				LED2_ON();
				Delay_ms(20);
				LED1_OFF();
			}
			if(RxData==3){
				LED2_OFF();
			}
		
			if(RxData==0 || RxData==1){
				OLED_ShowNum(1, 9, RxData, 1);	//显示串口接收的数据
			}
			if(RxData==2 || RxData==3){
				OLED_ShowNum(2, 9, RxData, 1);	//显示串口接收的数据
			}
		}
	}
}
