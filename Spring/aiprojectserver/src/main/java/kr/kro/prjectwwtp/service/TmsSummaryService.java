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
import kr.kro.prjectwwtp.domain.FlowSummary;
import kr.kro.prjectwwtp.domain.TmsSummary;
import kr.kro.prjectwwtp.persistence.FakeDateRepository;
import kr.kro.prjectwwtp.persistence.FlowSummaryRepository;
import kr.kro.prjectwwtp.persistence.TmsSummaryRepository;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class TmsSummaryService {
	private final TmsSummaryRepository tmsRepo;
	private final FlowSummaryRepository flowRepo;
	private final FakeDateRepository fakeDateRepo;
	
	@PostConstruct
	public void init() {
		TimeZone.setDefault(TimeZone.getTimeZone("Asia/Seoul"));
	}

	int checkNum = 2600;
	public List<Date> getFakeTmsDatesList() {
		List<Date> retList = new ArrayList<Date>();
		List<TmsSummary> summaries = tmsRepo.findAll();
		
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
	
	public List<Date> getFakeFlowDatesList() {
		List<Date> retList = new ArrayList<Date>();
		List<FlowSummary> summaries = flowRepo.findAll();
		
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
		// 등록된 값이 오늘 생성한 날짜면 그냥 사용
		if(fakeDate != null
				&& fakeDate.getToday().isAfter(LocalDateTime.now().withHour(0).withMinute(0))) {
			System.out.println("fakeDate.getFakeDate() : " + fakeDate.getTmsDate());
			return fakeDate.getTmsDate();
		}
		
		List<Date> fakeDates = getFakeTmsDatesList();
		Random rand = new Random();
		int idx = rand.nextInt(fakeDates.size());
		Date retDate = fakeDates.get(idx);
		LocalDateTime tmsTime = retDate.toInstant() 
								.atZone(ZoneId.systemDefault())
								.toLocalDateTime();
		
		fakeDates = getFakeFlowDatesList();
		idx = rand.nextInt(fakeDates.size());
		retDate = fakeDates.get(idx);
		LocalDateTime flowTime = retDate.toInstant() 
				.atZone(ZoneId.systemDefault())
				.toLocalDateTime();
		
		fakeDateRepo.save(FakeDate.builder()
				.today(LocalDateTime.now())
				.tmsDate(tmsTime)
				.flowDate(flowTime)
				.build());
		System.out.println("new tmsDate : " + tmsTime + ", " + flowTime);
		return tmsTime;
				
	}
}
