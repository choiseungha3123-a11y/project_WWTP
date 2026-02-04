package kr.kro.prjectwwtp.service;

import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Random;
import java.util.TimeZone;

import org.springframework.stereotype.Service;

import jakarta.annotation.PostConstruct;
import kr.kro.prjectwwtp.domain.FakeDate;
import kr.kro.prjectwwtp.domain.TmsSummary;
import kr.kro.prjectwwtp.persistence.FakeDateRepository;
import kr.kro.prjectwwtp.persistence.TmsSummaryRepository;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class TmsSummaryService {
	private final TmsSummaryRepository repo;
	private final FakeDateRepository fakeDateRepo;
	
	@PostConstruct
	public void init() {
		TimeZone.setDefault(TimeZone.getTimeZone("Asia/Seoul"));
	}
	
	public List<Date> getFakeDatesList() {
		List<Date> retList = new ArrayList<Date>();
		List<TmsSummary> summaries = repo.findAll();
		int checkNum = 2600;
		
		TmsSummary pre = null;
		for(TmsSummary summary : summaries) {
			if(pre == null) {
				pre = summary;
				continue;
			}
			if( pre.getCount() + summary.getCount() >= checkNum &&
					ChronoUnit.DAYS.between(pre.getTime().toInstant(), summary.getTime().toInstant()) == 1) {
				// 하루전 날짜와의 합계가 checkNum 이상인 경우
				retList.add(summary.getTime());
				}
			pre = summary;
		}
		return retList;
	}
	
	public LocalDateTime getFakeNow() {
		FakeDate fakeDate = fakeDateRepo.findFirstByOrderByTodayDesc();
		if(fakeDate != null) {
			return fakeDate.getFakeDate();
		}
		
		List<Date> fakeDates = getFakeDatesList();
		Random rand = new Random();
		int idx = rand.nextInt(fakeDates.size());
		Date retDate = fakeDates.get(idx);
		LocalDateTime time = retDate.toInstant() 
								.atZone(ZoneId.systemDefault())
								.toLocalDateTime();
		fakeDateRepo.save(FakeDate.builder()
				.today(LocalDateTime.now())
				.fakeDate(time)
				.build());
		return time;
				
	}
}
