SELECT "timestamp", timestamp_spot, "Capitl", "Centrl", "Dunwod", "Genese", "Hud Vl", "Longil", "Mhk Vl", "Millwd", "N.Y.C.", "North", "West", "NYISO"
	
	select date_part('year', timestamp) as year, avg(abs(f_load-h_load))/avg(h_load) as error from (select h.timestamp, "Integrated Load" h_load,"Capitl" f_load   FROM iso.nyiso_hist_load h
	join iso.nyiso_fst_load f on h.timestamp=f.timestamp
	where h."Name" ='CAPITL'  and h.timestamp between '2020-01-01' and '2024-12-31 23:00' 
	and f.timestamp - f.timestamp_spot <= interval '48 hours'
	order by timestamp_spot, h.timestamp) group by date_part('year', timestamp) order by year;


	SELECT date_part('year', timestamp) as year, --case when "Time Zone" ='EDT' then 1 else 0 end daylightsaving,
	stddev("Integrated Load") std , avg("Integrated Load") avg 
	FROM iso.nyiso_hist_load l where "Name"='CAPITL' and
	timestamp between '2018-07-01' and '2025-01-04' 
	
	group by date_part('year', timestamp) order by date_part('year',timestamp)
	--limit 10