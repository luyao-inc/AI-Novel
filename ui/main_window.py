# ui/main_window.py
# -*- coding: utf-8 -*-
import os
import threading
import logging
import traceback
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from .role_library import RoleLibrary
from llm_adapters import create_llm_adapter

from config_manager import load_config, save_config, test_llm_config, test_embedding_config
from utils import read_file, save_string_to_txt, clear_file_content
from tooltips import tooltips

from ui.context_menu import TextWidgetContextMenu
from ui.main_tab import build_main_tab, build_left_layout, build_right_layout
from ui.config_tab import build_config_tabview, load_config_btn, save_config_btn
from ui.novel_params_tab import build_novel_params_area, build_optional_buttons_area
from ui.generation_handlers import (
    generate_novel_architecture_ui,
    generate_chapter_blueprint_ui,
    generate_chapter_draft_ui,
    finalize_chapter_ui,
    do_consistency_check,
    import_knowledge_handler,
    clear_vectorstore_handler,
    show_plot_arcs_ui
)
from ui.setting_tab import build_setting_tab, load_novel_architecture, save_novel_architecture
from ui.directory_tab import build_directory_tab, load_chapter_blueprint, save_chapter_blueprint
from ui.character_tab import build_character_tab, load_character_state, save_character_state
from ui.summary_tab import build_summary_tab, load_global_summary, save_global_summary
from ui.chapters_tab import build_chapters_tab, refresh_chapters_list, on_chapter_selected, load_chapter_content, save_current_chapter, prev_chapter, next_chapter

import json

class NovelGeneratorGUI:
    """
    小说生成器的主GUI类，包含所有的界面布局、事件处理、与后端逻辑的交互等。
    """
    def __init__(self, master):
        self.master = master
        self.master.title("Novel Generator GUI")
        try:
            if os.path.exists("icon.ico"):
                self.master.iconbitmap("icon.ico")
        except Exception:
            pass
        self.master.geometry("1350x840")
        
        # --------------- 配置文件路径 ---------------
        self.config_path = "config.json"
        self.loaded_config = load_config(self.config_path)

        if self.loaded_config:
            last_llm = self.loaded_config.get("last_interface_format", "OpenAI")
            last_embedding = self.loaded_config.get("last_embedding_interface_format", "OpenAI")
        else:
            last_llm = "OpenAI"
            last_embedding = "OpenAI"

        if self.loaded_config and "llm_configs" in self.loaded_config and last_llm in self.loaded_config["llm_configs"]:
            llm_conf = self.loaded_config["llm_configs"][last_llm]
        else:
            llm_conf = {
                "api_key": "",
                "base_url": "https://api.openai.com/v1",
                "model_name": "gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": 8192,
                "timeout": 600
            }

        if self.loaded_config and "embedding_configs" in self.loaded_config and last_embedding in self.loaded_config["embedding_configs"]:
            emb_conf = self.loaded_config["embedding_configs"][last_embedding]
        else:
            emb_conf = {
                "api_key": "",
                "base_url": "https://api.openai.com/v1",
                "model_name": "text-embedding-ada-002",
                "retrieval_k": 4
            }

        # -- LLM通用参数 --
        self.api_key_var = ctk.StringVar(value=llm_conf.get("api_key", ""))
        self.base_url_var = ctk.StringVar(value=llm_conf.get("base_url", "https://api.openai.com/v1"))
        self.interface_format_var = ctk.StringVar(value=last_llm)
        self.model_name_var = ctk.StringVar(value=llm_conf.get("model_name", "gpt-4o-mini"))
        self.temperature_var = ctk.DoubleVar(value=llm_conf.get("temperature", 0.7))
        self.max_tokens_var = ctk.IntVar(value=llm_conf.get("max_tokens", 8192))
        self.timeout_var = ctk.IntVar(value=llm_conf.get("timeout", 600))

        # -- Embedding相关 --
        self.embedding_interface_format_var = ctk.StringVar(value=last_embedding)
        self.embedding_api_key_var = ctk.StringVar(value=emb_conf.get("api_key", ""))
        self.embedding_url_var = ctk.StringVar(value=emb_conf.get("base_url", "https://api.openai.com/v1"))
        self.embedding_model_name_var = ctk.StringVar(value=emb_conf.get("model_name", "text-embedding-ada-002"))
        self.embedding_retrieval_k_var = ctk.StringVar(value=str(emb_conf.get("retrieval_k", 4)))

        # -- 小说参数相关 --
        if self.loaded_config and "other_params" in self.loaded_config:
            op = self.loaded_config["other_params"]
            self.topic_default = op.get("topic", "")
            self.genre_var = ctk.StringVar(value=op.get("genre", "玄幻"))
            self.num_chapters_var = ctk.StringVar(value=str(op.get("num_chapters", 10)))
            self.word_number_var = ctk.StringVar(value=str(op.get("word_number", 3000)))
            self.filepath_var = ctk.StringVar(value=op.get("filepath", ""))
            self.chapter_num_var = ctk.StringVar(value=str(op.get("chapter_num", "1")))
            self.characters_involved_var = ctk.StringVar(value=op.get("characters_involved", ""))
            self.key_items_var = ctk.StringVar(value=op.get("key_items", ""))
            self.scene_location_var = ctk.StringVar(value=op.get("scene_location", ""))
            self.time_constraint_var = ctk.StringVar(value=op.get("time_constraint", ""))
            self.user_guidance_default = op.get("user_guidance", "")
        else:
            self.topic_default = ""
            self.genre_var = ctk.StringVar(value="玄幻")
            self.num_chapters_var = ctk.StringVar(value="10")
            self.word_number_var = ctk.StringVar(value="3000")
            self.filepath_var = ctk.StringVar(value="")
            self.chapter_num_var = ctk.StringVar(value="1")
            self.characters_involved_var = ctk.StringVar(value="")
            self.key_items_var = ctk.StringVar(value="")
            self.scene_location_var = ctk.StringVar(value="")
            self.time_constraint_var = ctk.StringVar(value="")
            self.user_guidance_default = ""

        # --------------- 整体Tab布局 ---------------
        self.tabview = ctk.CTkTabview(self.master)
        self.tabview.pack(fill="both", expand=True)

        # 创建各个标签页
        build_main_tab(self)
        build_config_tabview(self)
        build_novel_params_area(self, start_row=1)
        build_optional_buttons_area(self, start_row=2)
        build_setting_tab(self)
        build_directory_tab(self)
        build_character_tab(self)
        build_summary_tab(self)
        build_chapters_tab(self)

    # ----------------- 通用辅助函数 -----------------
    def show_tooltip(self, key: str):
        info_text = tooltips.get(key, "暂无说明")
        messagebox.showinfo("参数说明", info_text)

    def safe_get_int(self, var, default=1):
        try:
            val_str = str(var.get()).strip()
            return int(val_str)
        except:
            var.set(str(default))
            return default

    def safe_log(self, message):
        """线程安全的日志记录方法"""
        if hasattr(self, 'log_text'):
            # 如果是在GUI线程之外调用，使用after方法
            self.master.after(0, lambda: self._append_log(message))
        else:
            print(f"[LOG] {message}")

    def _append_log(self, message):
        """向日志文本框中追加消息"""
        if hasattr(self, 'log_text'):
            try:
                self.log_text.configure(state="normal")
                self.log_text.insert("end", f"{message}\n")
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
            except Exception as e:
                print(f"日志记录失败: {str(e)}\n消息: {message}")

    def disable_button_safe(self, btn):
        self.master.after(0, lambda: btn.configure(state="disabled"))

    def enable_button_safe(self, btn):
        self.master.after(0, lambda: btn.configure(state="normal"))

    def handle_exception(self, context: str, error_msg=None, traceback_str=None):
        if traceback_str is None:
            traceback_str = traceback.format_exc()
        
        full_message = f"{context}"
        if error_msg:
            full_message += f"\n错误: {error_msg}"
        full_message += f"\n{traceback_str}"
        
        logging.error(full_message)
        self.safe_log(full_message)

    def show_chapter_in_textbox(self, text: str):
        """将文本显示在章节编辑框中，并添加调试日志"""
        print(f"show_chapter_in_textbox被调用，显示文本长度:{len(text)}")
        print(f"显示前几个字符: {text[:50]}...")
        try:
            self.chapter_result.delete("0.0", "end")
            self.chapter_result.insert("0.0", text)
            self.chapter_result.see("end")
            print(f"✅ 文本已成功显示在chapter_result中")
            self.safe_log(f"✅ 文本已成功显示在编辑框中，长度：{len(text)}字")
        except Exception as e:
            error_msg = str(e)
            print(f"❌ 显示文本失败: {error_msg}")
            print(f"错误详情: {traceback.format_exc()}")
            self.safe_log(f"❌ 显示文本失败: {error_msg}")
            # 尝试备选方案
            try:
                self.chapter_result.configure(state="normal")  # 确保文本框可编辑
                self.chapter_result.delete("0.0", "end")
                self.chapter_result.insert("0.0", text)
                self.chapter_result.configure(state="normal")  # 保持可编辑状态
                print("✅ 使用备选方案成功显示文本")
                self.safe_log("✅ 使用备选方案成功显示文本")
            except Exception as e2:
                print(f"❌ 备选方案也失败: {str(e2)}")
                self.safe_log(f"❌ 备选方案也失败: {str(e2)}")
    
    def test_llm_config(self):
        """
        测试当前的LLM配置是否可用
        """
        print("测试LLM配置按钮被点击")
        self.safe_log("=========================================")
        self.safe_log("正在测试LLM配置...")
        
        # 获取测试按钮并更新状态
        test_btn = None
        for widget in self.master.winfo_children():
            if isinstance(widget, ctk.CTkButton) and widget.cget("text") == "测试LLM配置":
                test_btn = widget
                break
        
        if test_btn:
            test_btn.configure(text="测试中...", state="disabled")
            self.master.update_idletasks()  # 强制更新UI
        
        interface_format = self.interface_format_var.get().strip()
        api_key = self.api_key_var.get().strip()
        base_url = self.base_url_var.get().strip()
        model_name = self.model_name_var.get().strip()
        temperature = self.temperature_var.get()
        max_tokens = self.max_tokens_var.get()
        timeout = self.timeout_var.get()
        
        # 记录详细的配置信息
        self.safe_log(f"API Key: {api_key[:5]}...{api_key[-3:] if len(api_key) > 8 else ''}")
        self.safe_log(f"接口格式: {interface_format}")
        self.safe_log(f"基础URL: {base_url}")
        self.safe_log(f"模型名称: {model_name}")
        self.safe_log(f"温度: {temperature}")
        self.safe_log(f"最大Token: {max_tokens}")
        self.safe_log(f"超时: {timeout}秒")

        def on_test_complete():
            if test_btn:
                test_btn.configure(text="测试LLM配置", state="normal")
        
        # 定义回调函数在测试完成后恢复按钮状态
        def handle_exception_and_reset(context, error_msg=None, traceback_str=None):
            self.handle_exception(context, error_msg, traceback_str)
            on_test_complete()
        
        test_llm_config(
            interface_format=interface_format,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            log_func=self.safe_log,
            handle_exception_func=handle_exception_and_reset
        )
        
        # 设置延迟恢复按钮状态的计时器
        self.master.after(1000, lambda: self.master.after(timeout * 1000, on_test_complete))

    def test_embedding_config(self):
        """
        测试当前的Embedding配置是否可用
        """
        print("测试Embedding配置按钮被点击")
        self.safe_log("=========================================")
        self.safe_log("正在测试Embedding配置...")
        
        # 获取测试按钮并更新状态
        test_btn = None
        for widget in self.master.winfo_children():
            if isinstance(widget, ctk.CTkButton) and widget.cget("text") == "测试Embedding配置":
                test_btn = widget
                break
        
        if test_btn:
            test_btn.configure(text="测试中...", state="disabled")
            self.master.update_idletasks()  # 强制更新UI
        
        api_key = self.embedding_api_key_var.get().strip()
        base_url = self.embedding_url_var.get().strip()
        interface_format = self.embedding_interface_format_var.get().strip()
        model_name = self.embedding_model_name_var.get().strip()
        
        # 记录详细的配置信息
        self.safe_log(f"Embedding API Key: {api_key[:5]}...{api_key[-3:] if len(api_key) > 8 else ''}")
        self.safe_log(f"Embedding接口格式: {interface_format}")
        self.safe_log(f"Embedding基础URL: {base_url}")
        self.safe_log(f"Embedding模型名称: {model_name}")

        def on_test_complete():
            if test_btn:
                test_btn.configure(text="测试Embedding配置", state="normal")
        
        # 定义回调函数在测试完成后恢复按钮状态
        def handle_exception_and_reset(context, error_msg=None, traceback_str=None):
            self.handle_exception(context, error_msg, traceback_str)
            on_test_complete()
        
        test_embedding_config(
            api_key=api_key,
            base_url=base_url,
            interface_format=interface_format,
            model_name=model_name,
            log_func=self.safe_log,
            handle_exception_func=handle_exception_and_reset
        )
        
        # 设置延迟恢复按钮状态的计时器
        self.master.after(1000, lambda: self.master.after(30000, on_test_complete))
    
    def browse_folder(self):
        # 获取上次使用的路径作为初始目录
        initial_dir = self.filepath_var.get().strip()
        
        # 如果没有上次使用的路径，则使用用户主目录
        if not initial_dir or not os.path.exists(initial_dir):
            initial_dir = os.path.expanduser("~/Desktop")
        
        selected_dir = filedialog.askdirectory(initialdir=initial_dir)
        if selected_dir:
            self.filepath_var.set(selected_dir)
            
            # 将路径保存到配置文件中
            try:
                config = self.loaded_config if self.loaded_config else {}
                
                # 确保other_params存在
                if "other_params" not in config:
                    config["other_params"] = {}
                    
                # 更新路径
                config["other_params"]["filepath"] = selected_dir
                
                # 保存配置
                save_config(config, self.config_path)
                print(f"已将路径 {selected_dir} 保存到配置文件")
            except Exception as e:
                print(f"保存路径到配置文件时出错: {str(e)}")
                # 不影响正常使用，所以不显示错误消息

    def show_character_import_window(self):
        """显示角色导入窗口"""
        import_window = ctk.CTkToplevel(self.master)
        import_window.title("导入角色信息")
        import_window.geometry("600x500")
        import_window.transient(self.master)  # 设置为父窗口的临时窗口
        import_window.grab_set()  # 保持窗口在顶层
        
        # 主容器
        main_frame = ctk.CTkFrame(import_window)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 滚动容器
        scroll_frame = ctk.CTkScrollableFrame(main_frame)
        scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 获取角色库路径
        role_lib_path = os.path.join(self.filepath_var.get().strip(), "角色库")
        self.selected_roles = []  # 存储选中的角色名称
        
        # 动态加载角色分类
        if os.path.exists(role_lib_path):
            # 配置网格布局参数
            scroll_frame.columnconfigure(0, weight=1)
            max_roles_per_row = 4
            current_row = 0
            
            for category in os.listdir(role_lib_path):
                category_path = os.path.join(role_lib_path, category)
                if os.path.isdir(category_path):
                    # 创建分类容器
                    category_frame = ctk.CTkFrame(scroll_frame)
                    category_frame.grid(row=current_row, column=0, sticky="w", pady=(10,5), padx=5)
                    
                    # 添加分类标签
                    category_label = ctk.CTkLabel(category_frame, text=f"【{category}】", 
                                                font=("Microsoft YaHei", 12, "bold"))
                    category_label.grid(row=0, column=0, padx=(0,10), sticky="w")
                    
                    # 初始化角色排列参数
                    role_count = 0
                    row_num = 0
                    col_num = 1  # 从第1列开始（第0列是分类标签）
                    
                    # 添加角色复选框
                    for role_file in os.listdir(category_path):
                        if role_file.endswith(".txt"):
                            role_name = os.path.splitext(role_file)[0]
                            if not any(name == role_name for _, name in self.selected_roles):
                                chk = ctk.CTkCheckBox(category_frame, text=role_name)
                                chk.grid(row=row_num, column=col_num, padx=5, pady=2, sticky="w")
                                self.selected_roles.append((chk, role_name))
                                
                                # 更新行列位置
                                role_count += 1
                                col_num += 1
                                if col_num > max_roles_per_row:
                                    col_num = 1
                                    row_num += 1
                    
                    # 如果没有角色，调整分类标签占满整行
                    if role_count == 0:
                        category_label.grid(columnspan=max_roles_per_row+1, sticky="w")
                    
                    # 更新主布局的行号
                    current_row += 1
                    
                    # 添加分隔线
                    separator = ctk.CTkFrame(scroll_frame, height=1, fg_color="gray")
                    separator.grid(row=current_row, column=0, sticky="ew", pady=5)
                    current_row += 1
        
        # 底部按钮框架
        btn_frame = ctk.CTkFrame(main_frame)
        btn_frame.pack(fill="x", pady=10)
        
        # 选择按钮
        def confirm_selection():
            selected = [name for chk, name in self.selected_roles if chk.get() == 1]
            self.char_inv_text.delete("0.0", "end")
            self.char_inv_text.insert("0.0", ", ".join(selected))
            import_window.destroy()
            
        btn_confirm = ctk.CTkButton(btn_frame, text="选择", command=confirm_selection)
        btn_confirm.pack(side="left", padx=20)
        
        # 取消按钮
        btn_cancel = ctk.CTkButton(btn_frame, text="取消", command=import_window.destroy)
        btn_cancel.pack(side="right", padx=20)

    def show_role_library(self):
        save_path = self.filepath_var.get().strip()
        if not save_path:
            messagebox.showwarning("警告", "请先设置保存路径")
            return
        
        # 初始化LLM适配器
        llm_adapter = create_llm_adapter(
            interface_format=self.interface_format_var.get(),
            base_url=self.base_url_var.get(),
            model_name=self.model_name_var.get(),
            api_key=self.api_key_var.get(),
            temperature=self.temperature_var.get(),
            max_tokens=self.max_tokens_var.get(),
            timeout=self.timeout_var.get()
        )
        
        # 传递LLM适配器实例到角色库
        if hasattr(self, '_role_lib'):
            if self._role_lib.window and self._role_lib.window.winfo_exists():
                self._role_lib.window.destroy()
        
        self._role_lib = RoleLibrary(self.master, save_path, llm_adapter)  # 新增参数

    # ----------------- 将导入的各模块函数直接赋给类方法 -----------------
    generate_novel_architecture_ui = generate_novel_architecture_ui
    generate_chapter_blueprint_ui = generate_chapter_blueprint_ui
    generate_chapter_draft_ui = generate_chapter_draft_ui
    finalize_chapter_ui = finalize_chapter_ui
    do_consistency_check = do_consistency_check
    import_knowledge_handler = import_knowledge_handler
    clear_vectorstore_handler = clear_vectorstore_handler
    show_plot_arcs_ui = show_plot_arcs_ui
    load_config_btn = load_config_btn
    save_config_btn = save_config_btn
    load_novel_architecture = load_novel_architecture
    save_novel_architecture = save_novel_architecture
    load_chapter_blueprint = load_chapter_blueprint
    save_chapter_blueprint = save_chapter_blueprint
    load_character_state = load_character_state
    save_character_state = save_character_state
    load_global_summary = load_global_summary
    save_global_summary = save_global_summary
    refresh_chapters_list = refresh_chapters_list
    on_chapter_selected = on_chapter_selected
    save_current_chapter = save_current_chapter
    prev_chapter = prev_chapter
    next_chapter = next_chapter
    test_llm_config = test_llm_config
    test_embedding_config = test_embedding_config
    browse_folder = browse_folder

    def create_menu(self):
        """创建主界面菜单栏"""
        menu_bar = tk.Menu(self.master)
        self.master.config(menu=menu_bar)
        
        # 文件菜单
        file_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="选择工作目录", command=self.browse_folder)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.master.quit)
        
        # 生成菜单
        generate_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="生成", menu=generate_menu)
        generate_menu.add_command(label="生成小说架构", command=self.generate_novel_architecture_ui)
        generate_menu.add_command(label="生成章节规划", command=self.generate_chapter_blueprint_ui)
        generate_menu.add_command(label="生成章节草稿", command=self.generate_chapter_draft_ui)
        generate_menu.add_command(label="定稿当前章节", command=self.finalize_chapter_ui)
        
        # 工具菜单
        tools_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="工具", menu=tools_menu)
        tools_menu.add_command(label="角色库", command=self.show_role_library)
        tools_menu.add_command(label="检查一致性", command=self.do_consistency_check)
        tools_menu.add_command(label="导入知识", command=self.import_knowledge_handler)
        tools_menu.add_command(label="清空知识库", command=self.clear_vectorstore_handler)
        
        # 帮助菜单
        help_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="关于", command=lambda: messagebox.showinfo("关于", "人工智能小说生成器 v1.0\n作者: AI编程助手"))

    def create_log_window(self):
        """创建日志窗口"""
        # 创建日志窗口
        self.log_window = ctk.CTkToplevel(self.master)
        self.log_window.title("日志窗口")
        self.log_window.geometry("800x400")
        self.log_window.withdraw()  # 初始隐藏
        
        # 创建日志文本框
        self.log_text = ctk.CTkTextbox(self.log_window, wrap="word", font=("Consolas", 11))
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.log_text.configure(state="disabled")
        
        # 创建关闭按钮
        close_btn = ctk.CTkButton(self.log_window, text="关闭", command=self.log_window.withdraw)
        close_btn.pack(pady=10)
        
        # 设置窗口关闭事件
        self.log_window.protocol("WM_DELETE_WINDOW", self.log_window.withdraw)

    def create_gui(self):
        """创建主界面布局"""
        # 主要布局容器
        main_container = ctk.CTkFrame(self.master)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 设置整体的布局比例
        main_container.columnconfigure(0, weight=3)  # 左侧占比3
        main_container.columnconfigure(1, weight=7)  # 右侧占比7
        main_container.rowconfigure(0, weight=1)
        
        # 创建左侧区域 - 包含控制面板和参数设置
        left_frame = ctk.CTkFrame(main_container)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        build_left_layout(self, left_frame)
        
        # 创建右侧区域 - 包含文本编辑器和结果显示
        right_frame = ctk.CTkFrame(main_container)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        build_right_layout(self, right_frame)
