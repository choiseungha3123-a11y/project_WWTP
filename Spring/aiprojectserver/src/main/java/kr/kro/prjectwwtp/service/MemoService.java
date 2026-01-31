package kr.kro.prjectwwtp.service;

import java.time.LocalDateTime;
import java.util.Optional;

import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;

import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.Memo;
import kr.kro.prjectwwtp.domain.MemoLog;
import kr.kro.prjectwwtp.domain.PageDTO;
import kr.kro.prjectwwtp.persistence.MemoLogRepository;
import kr.kro.prjectwwtp.persistence.MemoRepository;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class MemoService {
	private final MemoRepository memoRepo;
	private final MemoLogRepository logRepo;
	
	public PageDTO<Memo> findByDisableMemberIsNull(Member member, Pageable pageable) {
		logRepo.save(MemoLog.builder()
						.type("list")
						.page(pageable.getPageNumber())
						.count(pageable.getPageSize())
						.member(member)
						.build());
		return new PageDTO<>(memoRepo.findByDisableMemberIsNull(pageable));
	}
	
	public PageDTO<Memo> findByDisableMemberIsNotNull(Member member, Pageable pageable) {
		logRepo.save(MemoLog.builder()
						.type("oldlist")
						.page(pageable.getPageNumber())
						.count(pageable.getPageSize())
						.member(member)
						.build());
		return new PageDTO<>(memoRepo.findByDisableMemberIsNotNull(pageable));
	}
	
	public void addMemo(Member member, String content) {
		Memo newMemo = Memo.builder()
				.content(content)
				.createMember(member)
				.build();
		memoRepo.save(newMemo);
		logRepo.save(MemoLog.builder()
						.type("create")
						.memoNo(newMemo.getMemoNo())
						.currentContent(content)
						.member(member)
						.build());
	}
	
	public void modifyMemo(Member member, long memoNo, String content) throws Exception {
		Optional<Memo> opt = memoRepo.findByMemoNoAndDisableMemberIsNull(memoNo);
		if(opt.isEmpty())
			throw new Exception("memoNo가 올바르지 않습니다.");
		Memo modifyMemo = opt.get();
		logRepo.save(MemoLog.builder()
						.type("modify")
						.memoNo(memoNo)
						.preContent(modifyMemo.getContent())
						.currentContent(content)
						.member(member)
						.build());
		modifyMemo.setContent(content);
		modifyMemo.setModifyMember(member);
		memoRepo.save(modifyMemo);
	}
	
	public void disableMemo(Member member, long memoNo) throws Exception {
		Optional<Memo> opt = memoRepo.findByMemoNoAndDisableMemberIsNull(memoNo);
		if(opt.isEmpty())
			throw new Exception("memoNo가 올바르지 않습니다.");
		Memo disableMemo = opt.get();
		logRepo.save(MemoLog.builder()
				.type("disable")
				.memoNo(memoNo)
				.currentContent(disableMemo.getContent())
				.member(member)
				.build());
		disableMemo.setDisableMember(member);
		disableMemo.setDisableTime(LocalDateTime.now());
		memoRepo.save(disableMemo);
	}

}
