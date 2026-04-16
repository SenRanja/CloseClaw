# encoding=utf-8
import re
from datetime import datetime
from pprint import pprint
from lxml import etree

from selenium import webdriver
from selenium.common import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# 初始化 Selenium WebDriver（假设你已经启动了浏览器并加载页面）
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC

# ========= SQLITE 数据库处理 =========
import sqlite3
# 1. 连接数据库（没有就自动创建）
conn = sqlite3.connect("imdb.db")
# 2. 创建游标
cursor = conn.cursor()
# 3. 执行 SQL
cursor.execute("""
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_text TEXT,
    rating INTEGER,
    UNIQUE(review_text, rating)
)
""")

# ========= CHROME DRIVER =========
executable_path = './chromedriver.exe'

chrome_options = Options()
# 以下 一个是直接启动chrome 一个是 接管chrome，二选一
# chrome_options.binary_location = r"./chrome-win64/chrome-win64/chrome.exe"
chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9229") # 如果接管现存chrome，流量不会走代理
# 启动chrome客户端命令： chrome.exe --remote-debugging-port=9229 --user-data-dir="D:\chrome-debug"

# # 禁用图片
# prefs = {
#     "profile.managed_default_content_settings.images": 2,
#     "profile.managed_default_content_settings.javascript": 2,
#     "profile.managed_default_content_settings.stylesheets": 2,
# "profile.managed_default_content_settings.fonts": 2,
# "profile.managed_default_content_settings.media_stream": 2,
# "profile.block_third_party_cookies": True,
#
# }
# chrome_options.add_experimental_option("prefs", prefs)
# # 关闭后台网络
# chrome_options.add_argument("--disable-background-networking")
# # 关闭扩展 & 同步
# chrome_options.add_argument("--disable-extensions")
# chrome_options.add_argument("--disable-sync")

# 代理
chrome_options.add_argument("--proxy-server=http://127.0.0.1:7777")
# 注意，目前单独命令行启动chrome的debug模式无法使用无头模式
# chrome_options.add_argument("--headless")  # 无头模式
chrome_options.add_argument("--mute-audio")  # 静音
chrome_options.add_argument("--disable-gpu")  # 不用GPU
# 绕过不必要的风险监测
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
# chrome_options.add_argument("--headless=new")
chrome_options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
# chrome_options.add_argument("--window-size=1920,1080")
# Linux/服务器建议加
# chrome_options.add_argument("--no-sandbox")
# chrome_options.add_argument("--disable-dev-shm-usage")

def main(executable_path):
    service = Service(executable_path=executable_path)
    # Start the browser
    browser = webdriver.Chrome(service=service, options=chrome_options)

    for movie_title_num in range(301750, 117117, -1):
        for err_i in range(2):
            try:
                # URL
                url = "https://www.imdb.com/title/tt0{MOVIE_TITLE_NUM}/reviews/".format(MOVIE_TITLE_NUM=movie_title_num)
                print("请求网页：" + url)
                browser.get(url)

                # Check if banned
                html = browser.page_source
                # print(html[:1000])
                warning = "Max challenge attempts exceeded. Please refresh the page to try again!"
                if warning in html:
                    print("被检查到，刷新！")
                    continue
                WebDriverWait(browser, 1).until(
                    EC.presence_of_all_elements_located((By.XPATH, "//article"))
                )
                articles = browser.find_elements(By.XPATH, "//article")

                if articles is None or len(articles) == 0:
                    break

                for article in articles:
                    try:
                        review_text = article.find_element(
                            By.XPATH, ".//div[1]/div[1]/div[3]/div/div/div"
                        ).text

                        review_rating = article.find_element(
                            By.XPATH, ".//div[1]/div[1]/div[1]/span/span[1]"
                        ).text

                        print("评论:", review_text)
                        print("评分:", review_rating)

                        if review_rating is not None and review_rating.isdigit() and len(review_text)>=20:
                            # 插入数据库
                            cursor.execute("""
                                INSERT OR IGNORE INTO reviews (review_text, rating)
                                VALUES (?, ?)
                                """, (review_text, review_rating))
                            # 提交
                            conn.commit()
                            print("INSERT INTO DB")
                        else:
                            print("NOT INSERT")
                    except NoSuchElementException as ne:
                        continue
                break

            except Exception as e_retry:
                print(e_retry)
                # 单个页面若出现报错，直接contine到下一页
                break

    # browser.quit()


if __name__ == "__main__":
    main(executable_path)