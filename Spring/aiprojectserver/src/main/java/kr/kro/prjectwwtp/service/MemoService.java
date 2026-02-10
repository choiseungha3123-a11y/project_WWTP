package kr.kro.prjectwwtp.service;

import java.time.LocalDateTime;
import java.util.Optional;

import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;

import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.Memo;
import kr.kro.prjectwwtp.domain.PageDTO;
import kr.kro.prjectwwtp.persistence.MemoRepository;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class MemoService {
	private final LogService logService;
	private final MemoRepository memoRepo;
	
	public PageDTO<Memo> findByDisableMemberIsNull(Member member, Pageable pageable) {
		logService.addMemoLog(member, "list", pageable.getPageNumber(), pageable.getPageSize(), 0, null, null);
		return new PageDTO<>(memoRepo.findByDisableMemberIsNull(pageable));
	}
	
	public PageDTO<Memo> findByDisableMemberIsNotNull(Member member, Pageable pageable) {
		logService.addMemoLog(member, "oldlist", pageable.getPageNumber(), pageable.getPageSize(), 0, null, null);
		return new PageDTO<>(memoRepo.findByDisableMemberIsNotNull(pageable));
	}
	
	public void addMemo(Member member, String content) {
		Memo newMemo = Memo.builder()
				.content(content)
				.createMember(member)
				.build();
		memoRepo.save(newMemo);
		logService.addMemoLog(member, "create", 0, 0, newMemo.getMemoNo(), content, null);
	}
	
	public void modifyMemo(Member member, long memoNo, String content) throws Exception {
		Optional<Memo> opt = memoRepo.findByMemoNoAndDisableMemberIsNull(memoNo);
		if(opt.isEmpty())
			throw new Exception("memoNo가 올바르지 않습니다.");
		Memo modifyMemo = opt.get();
		logService.addMemoLog(member, "modify", 0, 0, memoNo, content, modifyMemo.getContent());
		modifyMemo.setContent(content);
		modifyMemo.setModifyMember(member);
		memoRepo.save(modifyMemo);
	}
	
	public void disableMemo(Member member, long memoNo) throws Exception {
		Optional<Memo> opt = memoRepo.findByMemoNoAndDisableMemberIsNull(memoNo);
		if(opt.isEmpty())
			throw new Exception("memoNo가 올바르지 않습니다.");
		Memo disableMemo = opt.get();
		logService.addMemoLog(member, "disable", 0, 0, memoNo, disableMemo.getContent(), null);
		disableMemo.setDisableMember(member);
		disableMemo.setDisableTime(LocalDateTime.now());
		memoRepo.save(disableMemo);
	}
	
	public void deleteMemo(Member member, long memoNo) throws Exception {
		Optional<Memo> opt = memoRepo.findByMemoNoAndDisableMemberIsNull(memoNo);
		if(opt.isEmpty())
			throw new Exception("memoNo가 올바르지 않습니다.");
		Memo deleteMemo = opt.get();
		logService.addMemoLog(member, "delete", 0, 0, memoNo, deleteMemo.getContent(), null);
		memoRepo.delete(deleteMemo);
	}

}
