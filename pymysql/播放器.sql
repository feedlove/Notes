-- 创建音乐文件表
CREATE TABLE t_mp3(
ID INT PRIMARY KEY ,
mp3_name VARCHAR(200) not NULL  ,
mp3_file VARCHAR(200) NOT NULL,
CONSTRAINT u_test1 add UNIQUE (mp3_name)
);
CREATE TABLE t_user(
id INT PRIMARY KEY ,
username VARCHAR(200),NOT NULL  ,
password VARCHAR(200) NOT NULL
);
-- 用户播放列表
CREATE TABLE t_play_list(
id INT PRIMARY KEY ,
uid INT ,
mid int,
CONSTRAINT f_test1 ADD FOREIGN KEY (uid) REFERENCES t_user(id),
CONSTRAINT f_test2 ADD FOREIGN KEY (mid) REFERENCES t_play_list(id)
);