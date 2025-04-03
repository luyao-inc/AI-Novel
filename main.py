# main.py
# -*- coding: utf-8 -*-
import os
import sys
import platform
import customtkinter as ctk
from ui import NovelGeneratorGUI

def main():
    # 创建应用程序
    app = ctk.CTk()
    app.title("AI小说生成器")
    
    # 在macOS上设置必要的环境变量，确保tkinter在主线程上运行
    if platform.system() == 'Darwin':  # macOS
        os.environ['PYTHONUNBUFFERED'] = '1'  # 禁用输出缓冲
        os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'  # 禁用fork安全检查
        
        # 确保UI元素在主线程创建和操作
        from threading import Thread
        main_thread = Thread(target=lambda: None)
        main_thread._thread_id = 1  # 标记为主线程
        
        # 修复对话框问题
        import tkinter
        def fixed_showinfo(*args, **kwargs):
            app.lift()
            result = original_showinfo(*args, **kwargs)
            app.lift()
            return result
            
        original_showinfo = tkinter.messagebox.showinfo
        tkinter.messagebox.showinfo = fixed_showinfo
        
        # 设置其他消息框函数
        for name in ['showwarning', 'showerror', 'askquestion', 'askokcancel', 'askyesno']:
            if hasattr(tkinter.messagebox, name):
                original = getattr(tkinter.messagebox, name)
                def make_fixed(orig):
                    def fixed(*args, **kwargs):
                        app.lift()
                        result = orig(*args, **kwargs)
                        app.lift()
                        return result
                    return fixed
                setattr(tkinter.messagebox, name, make_fixed(original))
    
    # 设置应用程序图标（如果存在）
    try:
        icon_path = "icon.ico"
        if os.path.exists(icon_path):
            if platform.system() == "Darwin":  # macOS
                # macOS上需要特殊处理
                pass  # TK在macOS上处理图标的方式不同
            else:
                app.iconbitmap(icon_path)
    except Exception:
        pass  # 忽略图标设置错误
        
    # 配置窗口大小和位置
    app.geometry("1200x800")
    app.minsize(1000, 700)
    
    # 在显示前记录日志
    print("正在初始化GUI界面...")
    
    # 创建并显示GUI
    gui = NovelGeneratorGUI(app)
    
    # 启动消息循环
    print("GUI界面已准备就绪，开始主循环")
    app.mainloop()

if __name__ == "__main__":
    main()
