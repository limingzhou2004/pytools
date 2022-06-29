

create table calendar
(timestamp timestamp primary key,
holiday int,
daylightsaving int
)



create table load (
"timestamp" timestamp,
Name varchar(20),
"Time Zone" varchar(12),
"Integrated Load"  float,
primary key("timestamp", Name, "Time Zone")
)

