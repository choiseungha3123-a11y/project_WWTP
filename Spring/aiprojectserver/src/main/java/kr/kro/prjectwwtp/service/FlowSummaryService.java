package kr.kro.prjectwwtp.service;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.TimeZone;

import org.springframework.stereotype.Service;

import jakarta.annotation.PostConstruct;
import kr.kro.prjectwwtp.domain.FakeDate;
import kr.kro.prjectwwtp.domain.FlowSummary;
import kr.kro.prjectwwtp.persistence.FakeDateRepository;
import kr.kro.prjectwwtp.persistence.FlowSummaryRepository;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class FlowSummaryService {
	private final FlowSummaryRepository repo;
	private final FakeDateRepository fakeDateRepo;
	
	@PostConstruct
	public void init() {
		TimeZone.setDefault(TimeZone.getTimeZone("Asia/Seoul"));
	}
	
	public List<Date> getFakeDatesList() {
		List<Date> retList = new ArrayList<Date>();
		List<FlowSummary> summaries = repo.findAll();
		int checkNum = 2600;
		
		FlowSummary pre = null;
		for(FlowSummary summary : summaries) {
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
		return fakeDate.getFlowDate();
	}
}
