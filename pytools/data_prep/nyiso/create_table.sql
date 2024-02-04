
CREATE TABLE IF NOT EXISTS iso.nyiso_hist_load
(
    "timestamp" timestamp with time zone,
    "Name" varchar(30) ,
    "Time Zone" varchar(3) ,
    "Integrated Load" float
	
	primary key("timestamp", "Name", "Time Zone")
)



create table if not exists iso.nyiso_fst_load(
timestamp_utc timestamp with time zone,
timestamp_spot timestamp with time zone,
	"Capitl" float,
	"Centrl" float,
	"Dunwod" float,
	"Genese" float,
	"Hud Vl" float,
	"Longil" float,
	"Mhk Vl" float,
	"Millwd" float,
	"N.Y.C." float,
	"North" float,
	"West" float,
	"NYISO" float,
	primary key(timestamp_utc, timestamp_spot)
)
