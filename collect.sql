CREATE TABLE origin_img (
  origin_img_id VARCHAR(12) COLLATE utf8_unicode_ci PRIMARY KEY,
  img varchar(4096) COLLATE utf8_unicode_ci DEFAULT NULL,
  img_time DATETIME COLLATE utf8_unicode_ci NOT NULL
);
CREATE TABLE face (
  face_id VARCHAR(12) COLLATE utf8_unicode_ci  PRIMARY KEY,
  origin_img_id varchar(12) COLLATE utf8_unicode_ci  NOT NULL,
  face_img varchar(4096) COLLATE utf8_unicode_ci DEFAULT NULL,
FOREIGN KEY(origin_img_id) REFERENCES origin_img(origin_img_id)  
);
CREATE TABLE people (
  people_id VARCHAR(12) PRIMARY KEY
);
CREATE TABLE NTR (
  NTR_id VARCHAR(12) PRIMARY KEY
);
CREATE TABLE people_face_id(
	people_face_id int auto_increment primary key,
    people_id VARCHAR(12),
    face_id VARCHAR(12) COLLATE utf8_unicode_ci ,
    FOREIGN KEY(people_id) REFERENCES people(people_id),
    FOREIGN KEY(face_id) REFERENCES face(face_id)  
);
CREATE TABLE NTR_face_id(
	NTR_face_id int auto_increment primary key,
    NTR_id VARCHAR(12) NOT NULL,
    face_id VARCHAR(12) COLLATE utf8_unicode_ci,
    FOREIGN KEY(NTR_id) REFERENCES NTR(NTR_id),
    FOREIGN KEY(face_id) REFERENCES face(face_id)  
);