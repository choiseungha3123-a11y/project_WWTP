package kr.kro.prjectwwtp.service;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

import org.springframework.stereotype.Service;

import kr.kro.prjectwwtp.domain.Weather;
import kr.kro.prjectwwtp.persistence.WeatherRepository;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class WeatherService {
	private final WeatherRepository weatherRepo;
	
	public Weather findById(long id) {
		Optional<Weather> opt = weatherRepo.findById(id);
		if(opt.isEmpty())
			return null;
		return opt.get();
	}
	
	public Weather findFirstByStnOrderByDataNoDesc(int stn) {
		return weatherRepo.findFirstByStnOrderByDataNoDesc(stn);
	}
	
	public void saveWeatherList(List<Weather> list) {
		weatherRepo.saveAll(list);
	}
	
	public void deleteWeatherList(List<Weather> list) {
		weatherRepo.deleteAll(list);
	}
	
	public List<Weather> findByLogTimeBetween(LocalDateTime start, LocalDateTime end) {
		return weatherRepo.findByLogTimeBetween(start, end);
	}
	
	public List<Weather> findByStnAndLogTimeBetween(int stn, LocalDateTime start, LocalDateTime end) {
		return weatherRepo.findByStnAndLogTimeBetween(stn, start, end);
	}
	
	public void modifyWeahter(Weather data, double ta, double rn15m, double rn60m, double rn12h, double rnday, double hm, double td) {
		data.setTa(ta);
		data.setRn15m(rn15m);
		data.setRn60m(rn60m);
		data.setRn12h(rn12h);
		data.setRnday(rnday);
		data.setHm(hm);
		data.setTd(td);
		weatherRepo.save(data);
	}

}
